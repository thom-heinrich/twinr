# CHANGELOG: 2026-03-30
# BUG-1: Removed exclusive read locking and full-file reparsing on get()/load_all(); reads are now point queries on SQLite.
# BUG-2: Removed full-store double rewrites (primary + .bak) on every save; writes are now transactional page updates with bounded logical size checks.
# BUG-3: Added explicit store identity/schema version validation plus recovery for corrupt or missing primaries via SQLite backup or legacy JSON migration.
# BUG-4: Removed JSON-function CHECK constraints from the schema and added a portable rebuild path so
# BUG-4: trusted_schema=OFF still works on Pi SQLite builds where json_valid/json_type are not innocuous.
# SEC-1: Replaced legacy JSON read check-then-open with O_NOFOLLOW imports, tightened the final store directory to 0700, and forced owner-only perms on DB/backup/lock/sidecars.
# SEC-2: Hardened SQLite connections with trusted_schema=OFF, defensive mode, trigger/view disablement, ATTACH/SQL-length limits, secure_delete, and WAL truncation after writes.
# IMP-1: Frontier upgrade from ad-hoc JSON + flock to SQLite WAL + STRICT schema + application_id/user_version + point lookups + online backup.
# IMP-2: Added seamless migration from legacy integrations.json / integrations.json.bak into integrations.sqlite3.
# IMP-3: Canonical store path is now integrations.sqlite3; TwinrIntegrationStore still accepts a legacy .json path and migrates it automatically.

"""Persist Twinr integration configuration on disk.

This module owns the bounded, atomic, SQLite-backed store used by runtime and
web surfaces to read and update managed integration configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
import errno
import json
import logging
import os
import sqlite3
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar
from urllib.parse import quote

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]


_STORE_VERSION = 3
_APPLICATION_ID = 0x54574E52  # 'TWNR'
_SECURE_FILE_MODE = 0o600
_SECURE_DIR_MODE = 0o700
_MAX_STORE_BYTES = 1024 * 1024
_BUSY_TIMEOUT_MS = 5_000

logger = logging.getLogger(__name__)


def integration_store_path(project_root: str | Path) -> Path:
    """Return the canonical SQLite store path below one project root."""

    # BREAKING: The canonical store moved from integrations.json to integrations.sqlite3.
    return Path(project_root).resolve() / "artifacts" / "stores" / "integrations" / "integrations.sqlite3"


def legacy_integration_store_path(project_root: str | Path) -> Path:
    """Return the legacy JSON store path used for one-time migration."""

    return Path(project_root).resolve() / "artifacts" / "stores" / "integrations" / "integrations.json"


class IntegrationStoreError(RuntimeError):
    """Raised when the integration store cannot be safely read or written."""


class _PortableSchemaUpgradeRequired(IntegrationStoreError):
    """Raised when the store must be rebuilt into the portable schema."""


def _default_payload() -> dict[str, object]:
    """Return a fresh empty payload for a new legacy integration store."""

    return {"version": 1, "integrations": {}}


def _coerce_enabled(value: object) -> bool:
    """Parse persisted enabled flags into a safe boolean."""

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

    candidate = str(value).strip()
    if not candidate:
        raise ValueError("integration_id must not be empty")
    if any(ord(character) < 32 for character in candidate):
        raise ValueError("integration_id must not contain control characters")
    return candidate


def _normalize_settings(value: object) -> dict[str, str]:
    """Normalize one settings mapping to a JSON-safe string map."""

    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _schema_uses_portability_blocking_json_checks(schema_sql: object) -> bool:
    """Return whether one table DDL embeds JSON functions that break trusted-schema-safe SQLite."""

    normalized = str(schema_sql or "").casefold()
    return "json_valid(" in normalized or "json_type(" in normalized


def _sqlite_uri(path: Path, *, mode: str) -> str:
    """Build a safe SQLite file URI for one absolute path."""

    return f"file:{quote(path.as_posix(), safe='/')}?mode={mode}"


@dataclass(frozen=True, slots=True)
class ManagedIntegrationConfig:
    """Store one integration's enabled flag, settings, and update timestamp."""

    integration_id: str
    enabled: bool = False
    settings: dict[str, str] = field(default_factory=dict)
    updated_at: str | None = None

    @classmethod
    def from_dict(cls, integration_id: str, payload: dict[str, object]) -> "ManagedIntegrationConfig":
        """Build a normalized config record from persisted mapping data."""

        normalized_integration_id = _normalize_integration_id(integration_id)
        return cls(
            integration_id=normalized_integration_id,
            enabled=_coerce_enabled(payload.get("enabled", False)),
            settings=_normalize_settings(payload.get("settings", {})),
            updated_at=_normalize_updated_at(payload.get("updated_at")),
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ManagedIntegrationConfig":
        """Build one config record from an SQLite row."""

        try:
            settings_payload = json.loads(row["settings_json"])
        except (TypeError, ValueError):
            settings_payload = {}
        return cls(
            integration_id=_normalize_integration_id(row["integration_id"]),
            enabled=bool(row["enabled"]),
            settings=_normalize_settings(settings_payload),
            updated_at=_normalize_updated_at(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the config record into JSON-compatible data."""

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
        """Initialize the store for one explicit store path."""

        normalized_path = Path(path).expanduser()
        if not normalized_path.is_absolute():
            normalized_path = Path.cwd() / normalized_path
        normalized_path = Path(os.path.abspath(normalized_path))

        if normalized_path.suffix.casefold() == ".json":
            self._legacy_json_path = normalized_path
            self.path = normalized_path.with_suffix(".sqlite3")
        else:
            self.path = normalized_path
            self._legacy_json_path = normalized_path.with_suffix(".json")

        self._backup_path = self.path.with_name(f"{self.path.name}.bak")
        self._bootstrap_lock_path = self.path.with_name(f"{self.path.name}.bootstrap.lock")
        self._integrity_fingerprint: tuple[int, int] | None = None

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrIntegrationStore":
        """Build the store that belongs to one Twinr project root."""

        return cls(integration_store_path(project_root))

    def load_all(self) -> dict[str, ManagedIntegrationConfig]:
        """Load all normalized integration records from disk."""

        self._ensure_database_ready(allow_create=False)
        if not self.path.exists():
            return {}

        try:
            with self._connect(read_only=True) as connection:
                rows = connection.execute(
                    """
                    SELECT integration_id, enabled, settings_json, updated_at
                    FROM integration_configs
                    ORDER BY integration_id
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise IntegrationStoreError(f"Failed to load integration store: {self.path}") from exc

        loaded: dict[str, ManagedIntegrationConfig] = {}
        for row in rows:
            try:
                managed = ManagedIntegrationConfig.from_row(row)
            except ValueError:
                logger.warning("Skipping malformed integration row in %s: %r", self.path, dict(row))
                continue
            loaded[managed.integration_id] = managed
        return loaded

    def get(self, integration_id: str) -> ManagedIntegrationConfig:
        """Return one config record or an empty default record."""

        normalized_integration_id = _normalize_integration_id(integration_id)
        self._ensure_database_ready(allow_create=False)
        if not self.path.exists():
            return ManagedIntegrationConfig(integration_id=normalized_integration_id)

        try:
            with self._connect(read_only=True) as connection:
                row = connection.execute(
                    """
                    SELECT integration_id, enabled, settings_json, updated_at
                    FROM integration_configs
                    WHERE integration_id = ?
                    """,
                    (normalized_integration_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise IntegrationStoreError(
                f"Failed to read integration {normalized_integration_id!r} from {self.path}"
            ) from exc

        if row is None:
            return ManagedIntegrationConfig(integration_id=normalized_integration_id)
        try:
            return ManagedIntegrationConfig.from_row(row)
        except ValueError:
            logger.warning(
                "Ignoring malformed integration row for %s in %s",
                normalized_integration_id,
                self.path,
            )
            return ManagedIntegrationConfig(integration_id=normalized_integration_id)

    def save(self, record: ManagedIntegrationConfig) -> ManagedIntegrationConfig:
        """Persist one config record and return the normalized saved copy."""

        raw_settings = record.settings if isinstance(record.settings, dict) else {}
        normalized_record = ManagedIntegrationConfig(
            integration_id=_normalize_integration_id(record.integration_id),
            enabled=_coerce_enabled(record.enabled),
            settings={str(key): str(value) for key, value in raw_settings.items()},
            updated_at=_normalize_updated_at(record.updated_at),
        )
        saved_record = ManagedIntegrationConfig(
            integration_id=normalized_record.integration_id,
            enabled=normalized_record.enabled,
            settings=dict(sorted(normalized_record.settings.items())),
            updated_at=datetime.now(UTC).isoformat(),
        )
        settings_json = json.dumps(saved_record.settings, sort_keys=True, ensure_ascii=False)

        self._ensure_database_ready(allow_create=True)

        try:
            with self._connect(read_only=False) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._enforce_logical_size_limit_in_transaction(
                    connection=connection,
                    integration_id=saved_record.integration_id,
                    settings_json=settings_json,
                    updated_at=saved_record.updated_at or "",
                )
                connection.execute(
                    """
                    INSERT INTO integration_configs (integration_id, enabled, settings_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(integration_id) DO UPDATE SET
                        enabled = excluded.enabled,
                        settings_json = excluded.settings_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        saved_record.integration_id,
                        1 if saved_record.enabled else 0,
                        settings_json,
                        saved_record.updated_at,
                    ),
                )
                connection.commit()
                self._post_write_housekeeping(connection)
                self._refresh_backup_from_source_best_effort(connection)
        except sqlite3.Error as exc:
            raise IntegrationStoreError(
                f"Failed to save integration {saved_record.integration_id!r} to {self.path}"
            ) from exc

        self._refresh_integrity_fingerprint()
        return saved_record

    def _ensure_database_ready(self, allow_create: bool) -> None:
        """Ensure the canonical SQLite store exists, is valid, and is migrated."""

        with self._bootstrap_lock(create_parent=allow_create):
            if allow_create:
                self._ensure_store_parent()
            else:
                self._ensure_no_symlink_ancestors(self.path.parent)

            if self.path.exists():
                if self._database_changed_since_integrity_check():
                    self._verify_or_recover_database_locked()
                return

            if self._backup_path.exists():
                try:
                    self._restore_from_backup_locked()
                    return
                except IntegrationStoreError as exc:
                    logger.warning("Backup restore failed for missing store %s: %s", self.path, exc)

            if self._legacy_json_path.exists():
                self._migrate_legacy_json_to_sqlite_locked()
                return

            if allow_create:
                self._initialize_database_locked()

    def _database_changed_since_integrity_check(self) -> bool:
        """Return True when the database file changed since the last successful verification."""

        try:
            stat_result = self.path.stat()
        except FileNotFoundError:
            return False
        current = (stat_result.st_mtime_ns, stat_result.st_size)
        return current != self._integrity_fingerprint

    def _refresh_integrity_fingerprint(self) -> None:
        """Cache the current file fingerprint after a successful verified operation."""

        try:
            stat_result = self.path.stat()
        except FileNotFoundError:
            self._integrity_fingerprint = None
            return
        self._integrity_fingerprint = (stat_result.st_mtime_ns, stat_result.st_size)

    def _connect(self, *, read_only: bool) -> sqlite3.Connection:
        """Open one configured SQLite connection to the canonical store."""

        self._ensure_no_symlink_ancestors(self.path.parent)
        if self.path.exists() and self.path.is_symlink():
            raise IntegrationStoreError(f"Refusing symlinked integration store database: {self.path}")

        mode = "ro" if read_only else "rwc"
        try:
            connection = sqlite3.connect(
                _sqlite_uri(self.path, mode=mode),
                uri=True,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise IntegrationStoreError(f"Failed to open integration store database: {self.path}") from exc

        connection.row_factory = sqlite3.Row
        try:
            self._configure_connection(connection, read_only=read_only)
        except Exception:
            connection.close()
            raise
        return connection

    def _configure_connection(self, connection: sqlite3.Connection, *, read_only: bool) -> None:
        """Apply concurrency, durability, and security configuration to one connection."""

        try:
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema=OFF")
            if read_only:
                connection.execute("PRAGMA query_only=ON")
            else:
                journal_mode_row = connection.execute("PRAGMA journal_mode=WAL").fetchone()
                journal_mode = str(journal_mode_row[0]).casefold() if journal_mode_row is not None else ""
                if journal_mode != "wal":
                    raise IntegrationStoreError(f"SQLite WAL mode is unavailable for {self.path}")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA secure_delete=ON")
                connection.execute("PRAGMA journal_size_limit=0")
        except sqlite3.Error as exc:
            raise IntegrationStoreError(f"Failed to configure SQLite integration store: {self.path}") from exc

        for config_name, value in (
            ("SQLITE_DBCONFIG_DEFENSIVE", True),
            ("SQLITE_DBCONFIG_TRUSTED_SCHEMA", False),
            ("SQLITE_DBCONFIG_ENABLE_TRIGGER", False),
            ("SQLITE_DBCONFIG_ENABLE_VIEW", False),
        ):
            opcode = getattr(sqlite3, config_name, None)
            if opcode is not None and hasattr(connection, "setconfig"):
                try:
                    connection.setconfig(opcode, value)
                except sqlite3.Error:
                    pass

        for limit_name, limit_value in (
            ("SQLITE_LIMIT_ATTACHED", 0),
            ("SQLITE_LIMIT_SQL_LENGTH", 8192),
            ("SQLITE_LIMIT_LENGTH", _MAX_STORE_BYTES * 2),
        ):
            category = getattr(sqlite3, limit_name, None)
            if category is not None and hasattr(connection, "setlimit"):
                try:
                    connection.setlimit(category, limit_value)
                except sqlite3.Error:
                    pass

    def _initialize_database_locked(self) -> None:
        """Create an empty canonical SQLite store."""

        with self._connect(read_only=False) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._apply_schema_locked(connection)
            connection.commit()
            self._post_write_housekeeping(connection)
            self._refresh_backup_from_source_locked_best_effort(connection)
        self._refresh_integrity_fingerprint()

    def _migrate_legacy_json_to_sqlite_locked(self) -> None:
        """Import the legacy JSON payload into the canonical SQLite store."""

        payload = self._load_legacy_payload()
        integrations = payload.get("integrations", {})
        rows: list[tuple[str, int, str, str]] = []

        if isinstance(integrations, dict):
            for integration_id, record in integrations.items():
                if not isinstance(record, dict):
                    continue
                try:
                    managed = ManagedIntegrationConfig.from_dict(str(integration_id), record)
                except ValueError:
                    logger.warning(
                        "Skipping invalid integration_id in legacy store %s: %r",
                        self._legacy_json_path,
                        integration_id,
                    )
                    continue
                rows.append(
                    (
                        managed.integration_id,
                        1 if managed.enabled else 0,
                        json.dumps(dict(sorted(managed.settings.items())), sort_keys=True, ensure_ascii=False),
                        managed.updated_at or datetime.now(UTC).isoformat(),
                    )
                )

        with self._connect(read_only=False) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._apply_schema_locked(connection)
            if rows:
                connection.executemany(
                    """
                    INSERT INTO integration_configs (integration_id, enabled, settings_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(integration_id) DO UPDATE SET
                        enabled = excluded.enabled,
                        settings_json = excluded.settings_json,
                        updated_at = excluded.updated_at
                    """,
                    rows,
                )
            connection.commit()
            self._post_write_housekeeping(connection)
            self._refresh_backup_from_source_locked_best_effort(connection)
        logger.info("Migrated legacy integration store %s to %s", self._legacy_json_path, self.path)
        self._refresh_integrity_fingerprint()

    def _apply_schema_locked(self, connection: sqlite3.Connection) -> None:
        """Create or upgrade the canonical schema metadata inside one write transaction."""

        connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS integration_configs (
                integration_id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
                settings_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            ) STRICT
            """
        )
        # Version 3 keeps the SQLite backend but removes schema-level JSON
        # functions so trusted_schema=OFF remains portable to older Pi builds.
        connection.execute(f"PRAGMA user_version={_STORE_VERSION}")

    def _verify_or_recover_database_locked(self) -> None:
        """Verify the canonical SQLite store and recover from backup or legacy JSON when needed."""

        try:
            with self._connect(read_only=False) as connection:
                self._verify_database_connection_locked(connection)
        except _PortableSchemaUpgradeRequired as primary_exc:
            logger.warning("Integration store schema upgrade required for %s: %s", self.path, primary_exc)
            try:
                self._rebuild_database_from_sqlite_source_locked(
                    self.path,
                    destination_path=self.path,
                    reason=str(primary_exc),
                )
                return
            except IntegrationStoreError as rebuild_exc:
                logger.warning("Portable schema rebuild failed for %s: %s", self.path, rebuild_exc)
                primary_exc = rebuild_exc
        except (IntegrationStoreError, sqlite3.Error) as primary_exc:
            if self.path.exists() and self._is_unsafe_json_schema_error(primary_exc):
                logger.warning(
                    "Integration store uses legacy JSON CHECK schema and needs a portable rebuild: %s",
                    self.path,
                )
                try:
                    self._rebuild_database_from_sqlite_source_locked(
                        self.path,
                        destination_path=self.path,
                        reason="legacy JSON CHECK schema is incompatible with trusted_schema=OFF",
                    )
                    return
                except IntegrationStoreError as rebuild_exc:
                    logger.warning("Portable schema rebuild failed for %s: %s", self.path, rebuild_exc)
                    primary_exc = rebuild_exc
            logger.warning("Integration store verification failed for %s: %s", self.path, primary_exc)
            if self._backup_path.exists():
                self._restore_from_backup_locked()
                return
            if self._legacy_json_path.exists():
                self._replace_with_legacy_json_locked()
                return
            raise IntegrationStoreError(
                f"Integration store is corrupt and has no usable recovery source: {self.path}"
            ) from primary_exc
        self._refresh_integrity_fingerprint()

    def _verify_database_connection_locked(self, connection: sqlite3.Connection) -> None:
        """Validate database identity, schema, version, and on-disk integrity."""

        application_id_row = connection.execute("PRAGMA application_id").fetchone()
        application_id = int(application_id_row[0]) if application_id_row is not None else 0
        if application_id not in {0, _APPLICATION_ID}:
            raise IntegrationStoreError(
                f"Refusing unrelated SQLite file with application_id={application_id}: {self.path}"
            )

        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='integration_configs'"
        ).fetchone()
        if table_exists is None:
            raise IntegrationStoreError(f"SQLite integration schema is missing integration_configs in {self.path}")

        schema_sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='integration_configs'"
        ).fetchone()
        schema_sql = str(schema_sql_row[0] or "") if schema_sql_row is not None else ""
        if _schema_uses_portability_blocking_json_checks(schema_sql):
            raise _PortableSchemaUpgradeRequired(
                f"Integration store schema still uses JSON CHECK constraints and must be rebuilt: {self.path}"
            )

        user_version_row = connection.execute("PRAGMA user_version").fetchone()
        user_version = int(user_version_row[0]) if user_version_row is not None else 0
        if user_version > _STORE_VERSION:
            raise IntegrationStoreError(
                f"Integration store schema version {user_version} is newer than supported {_STORE_VERSION}: {self.path}"
            )

        table_info_rows = connection.execute("PRAGMA table_info(integration_configs)").fetchall()
        declared_types = {str(row["name"]): str(row["type"]).upper() for row in table_info_rows}
        expected_types = {
            "integration_id": "TEXT",
            "enabled": "INTEGER",
            "settings_json": "TEXT",
            "updated_at": "TEXT",
        }
        for column_name, expected_type in expected_types.items():
            if declared_types.get(column_name) != expected_type:
                raise IntegrationStoreError(
                    f"Unexpected SQLite integration schema in {self.path}: {declared_types!r}"
                )

        if application_id == 0 or user_version < _STORE_VERSION:
            connection.execute("BEGIN IMMEDIATE")
            self._apply_schema_locked(connection)
            connection.commit()

        quick_check_row = connection.execute("PRAGMA quick_check").fetchone()
        quick_check = str(quick_check_row[0]).casefold() if quick_check_row is not None else ""
        if quick_check != "ok":
            raise IntegrationStoreError(f"SQLite integrity check failed for {self.path}: {quick_check!r}")

    def _restore_from_backup_locked(self) -> None:
        """Restore the canonical database from the last known-good SQLite backup."""

        temp_path: str | None = None
        try:
            backup_connection = self._connect_file(self._backup_path, read_only=False)
            try:
                self._verify_database_connection_locked(backup_connection)
                descriptor, temp_path = tempfile.mkstemp(
                    prefix=f".{self.path.name}.restore.",
                    suffix=".tmp",
                    dir=str(self.path.parent),
                )
                os.close(descriptor)
                os.unlink(temp_path)
                target_connection = self._connect_file(Path(temp_path), read_only=False)
                try:
                    target_connection.execute("BEGIN IMMEDIATE")
                    self._apply_schema_locked(target_connection)
                    target_connection.commit()
                    backup_connection.backup(target_connection)
                finally:
                        target_connection.close()
            finally:
                backup_connection.close()

            self._replace_database_file(Path(temp_path), self.path)
            logger.warning("Recovered integration store %s from backup %s", self.path, self._backup_path)
            self._refresh_integrity_fingerprint()
        except _PortableSchemaUpgradeRequired as exc:
            self._rebuild_database_from_sqlite_source_locked(
                self._backup_path,
                destination_path=self.path,
                reason=str(exc),
            )
            logger.warning(
                "Recovered integration store %s from backup %s via portable schema rebuild",
                self.path,
                self._backup_path,
            )
        except Exception as exc:
            if self._backup_path.exists() and self._is_unsafe_json_schema_error(exc):
                self._rebuild_database_from_sqlite_source_locked(
                    self._backup_path,
                    destination_path=self.path,
                    reason="legacy JSON CHECK schema in backup is incompatible with trusted_schema=OFF",
                )
                logger.warning(
                    "Recovered integration store %s from backup %s via portable schema rebuild",
                    self.path,
                    self._backup_path,
                )
                return
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise IntegrationStoreError(f"Failed to restore integration store from backup: {self._backup_path}") from exc

    def _replace_with_legacy_json_locked(self) -> None:
        """Rebuild the canonical database from the legacy JSON store."""

        self._remove_sqlite_sidecars(self.path)
        try:
            if self.path.exists():
                os.unlink(self.path)
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to remove corrupt integration store: {self.path}") from exc
        self._migrate_legacy_json_to_sqlite_locked()
        logger.warning("Recovered integration store %s from legacy JSON %s", self.path, self._legacy_json_path)

    def _connect_file(self, path: Path, *, read_only: bool) -> sqlite3.Connection:
        """Open one configured SQLite connection to an arbitrary store or backup file."""

        self._ensure_no_symlink_ancestors(path.parent)
        if path.exists() and path.is_symlink():
            raise IntegrationStoreError(f"Refusing symlinked integration store file: {path}")

        mode = "ro" if read_only else "rwc"
        try:
            connection = sqlite3.connect(
                _sqlite_uri(path, mode=mode),
                uri=True,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise IntegrationStoreError(f"Failed to open SQLite database: {path}") from exc

        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema=OFF")
            if read_only:
                connection.execute("PRAGMA query_only=ON")
            else:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA secure_delete=ON")
        except Exception:
            connection.close()
            raise
        return connection

    def _connect_recovery_source(self, path: Path) -> sqlite3.Connection:
        """Open one read-only SQLite connection that can salvage legacy schema variants."""

        self._ensure_no_symlink_ancestors(path.parent)
        if path.exists() and path.is_symlink():
            raise IntegrationStoreError(f"Refusing symlinked integration store file: {path}")

        try:
            connection = sqlite3.connect(
                _sqlite_uri(path, mode="ro"),
                uri=True,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise IntegrationStoreError(f"Failed to open SQLite recovery source: {path}") from exc

        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            connection.execute("PRAGMA trusted_schema=ON")
            connection.execute("PRAGMA query_only=ON")
        except Exception:
            connection.close()
            raise

        for config_name, value in (
            ("SQLITE_DBCONFIG_DEFENSIVE", True),
            ("SQLITE_DBCONFIG_TRUSTED_SCHEMA", True),
            ("SQLITE_DBCONFIG_ENABLE_TRIGGER", False),
            ("SQLITE_DBCONFIG_ENABLE_VIEW", False),
        ):
            opcode = getattr(sqlite3, config_name, None)
            if opcode is not None and hasattr(connection, "setconfig"):
                try:
                    connection.setconfig(opcode, value)
                except sqlite3.Error:
                    pass
        return connection

    def _refresh_backup_from_source_locked_best_effort(self, source_connection: sqlite3.Connection) -> None:
        """Refresh the SQLite backup while the bootstrap lock is already held."""

        try:
            self._refresh_backup_from_source_locked(source_connection)
        except IntegrationStoreError as exc:
            logger.warning("Failed to refresh integration store backup %s: %s", self._backup_path, exc)

    def _refresh_backup_from_source_best_effort(self, source_connection: sqlite3.Connection) -> None:
        """Refresh the SQLite backup without failing the already-committed primary save."""

        try:
            with self._bootstrap_lock(create_parent=True):
                self._refresh_backup_from_source_locked(source_connection)
        except IntegrationStoreError as exc:
            logger.warning("Failed to refresh integration store backup %s: %s", self._backup_path, exc)

    def _refresh_backup_from_source_locked(self, source_connection: sqlite3.Connection) -> None:
        """Copy the canonical database into the last known-good backup file."""

        temp_path: str | None = None
        try:
            try:
                source_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass

            descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{self._backup_path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
            )
            os.close(descriptor)
            os.unlink(temp_path)

            target_connection = self._connect_file(Path(temp_path), read_only=False)
            try:
                target_connection.execute("BEGIN IMMEDIATE")
                self._apply_schema_locked(target_connection)
                target_connection.commit()
                source_connection.backup(target_connection)
                try:
                    target_connection.execute("PRAGMA journal_mode=DELETE")
                except sqlite3.Error:
                    pass
            finally:
                target_connection.close()

            self._replace_database_file(Path(temp_path), self._backup_path)
        except Exception as exc:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise IntegrationStoreError(
                f"Failed to refresh SQLite integration store backup: {self._backup_path}"
            ) from exc

    def _is_unsafe_json_schema_error(self, exc: object) -> bool:
        """Return whether one SQLite failure is caused by legacy JSON CHECK schema under trusted_schema=OFF."""

        pending: list[object] = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if current is None:
                continue
            marker = id(current)
            if marker in seen:
                continue
            seen.add(marker)

            normalized = str(current).casefold()
            if (
                "unsafe use of json_valid()" in normalized
                or "unsafe use of json_type()" in normalized
                or (
                    "malformed database schema" in normalized
                    and ("json_valid()" in normalized or "json_type()" in normalized)
                )
            ):
                return True

            if isinstance(current, BaseException):
                pending.append(current.__cause__)
                pending.append(current.__context__)
        return False

    def _normalized_rows_from_sqlite_source_locked(self, source_path: Path) -> list[tuple[str, int, str, str]]:
        """Load normalized rows from one existing SQLite store using the portable recovery connection."""

        with self._connect_recovery_source(source_path) as connection:
            application_id_row = connection.execute("PRAGMA application_id").fetchone()
            application_id = int(application_id_row[0]) if application_id_row is not None else 0
            if application_id not in {0, _APPLICATION_ID}:
                raise IntegrationStoreError(
                    f"Refusing unrelated SQLite file with application_id={application_id}: {source_path}"
                )

            table_exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='integration_configs'"
            ).fetchone()
            if table_exists is None:
                raise IntegrationStoreError(f"SQLite integration schema is missing integration_configs in {source_path}")

            rows = connection.execute(
                """
                SELECT integration_id, enabled, settings_json, updated_at
                FROM integration_configs
                ORDER BY integration_id
                """
            ).fetchall()

        normalized_rows: list[tuple[str, int, str, str]] = []
        for row in rows:
            try:
                managed = ManagedIntegrationConfig.from_row(row)
            except ValueError:
                logger.warning("Skipping malformed integration row while rebuilding %s: %r", source_path, dict(row))
                continue
            normalized_rows.append(
                (
                    managed.integration_id,
                    1 if managed.enabled else 0,
                    json.dumps(dict(sorted(managed.settings.items())), sort_keys=True, ensure_ascii=False),
                    managed.updated_at or datetime.now(UTC).isoformat(),
                )
            )
        return normalized_rows

    def _rebuild_database_from_sqlite_source_locked(
        self,
        source_path: Path,
        *,
        destination_path: Path,
        reason: str,
    ) -> None:
        """Rewrite one SQLite store into the portable schema while preserving normalized row content."""

        rows = self._normalized_rows_from_sqlite_source_locked(source_path)
        temp_path: str | None = None
        try:
            descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{destination_path.name}.portable.",
                suffix=".tmp",
                dir=str(destination_path.parent),
            )
            os.close(descriptor)
            os.unlink(temp_path)

            target_connection = self._connect_file(Path(temp_path), read_only=False)
            try:
                target_connection.execute("BEGIN IMMEDIATE")
                self._apply_schema_locked(target_connection)
                if rows:
                    target_connection.executemany(
                        """
                        INSERT INTO integration_configs (integration_id, enabled, settings_json, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(integration_id) DO UPDATE SET
                            enabled = excluded.enabled,
                            settings_json = excluded.settings_json,
                            updated_at = excluded.updated_at
                        """,
                        rows,
                    )
                target_connection.commit()
                self._post_write_housekeeping(target_connection)
            finally:
                target_connection.close()

            self._replace_database_file(Path(temp_path), destination_path)
            logger.warning(
                "Rebuilt SQLite integration store %s from %s using portable schema: %s",
                destination_path,
                source_path,
                reason,
            )
            if destination_path == self.path:
                with self._connect(read_only=False) as connection:
                    self._refresh_backup_from_source_locked_best_effort(connection)
                self._refresh_integrity_fingerprint()
        except Exception as exc:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            raise IntegrationStoreError(
                f"Failed to rebuild SQLite integration store into portable schema: {destination_path}"
            ) from exc

    def _post_write_housekeeping(self, connection: sqlite3.Connection) -> None:
        """Trim WAL residue and lock down runtime-created files after one write transaction."""

        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self._chmod_runtime_files()

    def _chmod_runtime_files(self) -> None:
        """Force owner-only permissions on the database, backup, lock, and SQLite sidecars."""

        for candidate in (
            self.path,
            self._backup_path,
            self._bootstrap_lock_path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
            self._backup_path.with_name(f"{self._backup_path.name}-wal"),
            self._backup_path.with_name(f"{self._backup_path.name}-shm"),
        ):
            try:
                if candidate.exists():
                    os.chmod(candidate, _SECURE_FILE_MODE)
            except OSError:
                continue

    def _enforce_logical_size_limit_in_transaction(
        self,
        *,
        connection: sqlite3.Connection,
        integration_id: str,
        settings_json: str,
        updated_at: str,
    ) -> None:
        """Enforce the bounded logical payload size inside the active write transaction."""

        row = connection.execute(
            """
            SELECT COALESCE(
                SUM(LENGTH(integration_id) + LENGTH(settings_json) + LENGTH(updated_at) + 8),
                0
            )
            FROM integration_configs
            WHERE integration_id <> ?
            """,
            (integration_id,),
        ).fetchone()
        current_total = int(row[0]) if row is not None and row[0] is not None else 0
        projected_total = (
            current_total
            + len(integration_id.encode("utf-8"))
            + len(settings_json.encode("utf-8"))
            + len(updated_at.encode("utf-8"))
            + 8
        )
        if projected_total > _MAX_STORE_BYTES:
            raise IntegrationStoreError(
                f"Integration store payload exceeds the {_MAX_STORE_BYTES}-byte safety limit"
            )

    def _load_legacy_payload(self) -> dict[str, object]:
        """Load the legacy JSON payload, falling back to its .bak copy when necessary."""

        primary_state, primary_payload = self._load_json_payload_file(self._legacy_json_path)
        if primary_state == "ok" and primary_payload is not None:
            return primary_payload

        backup_path = self._legacy_json_path.with_name(f"{self._legacy_json_path.name}.bak")
        backup_state, backup_payload = self._load_json_payload_file(backup_path)
        if backup_state == "ok" and backup_payload is not None:
            logger.warning("Recovered legacy integration store %s from backup %s", self._legacy_json_path, backup_path)
            return backup_payload

        if primary_state == "missing":
            return _default_payload()
        raise IntegrationStoreError(f"Legacy integration store cannot be loaded safely: {self._legacy_json_path}")

    def _load_json_payload_file(self, path: Path) -> tuple[str, dict[str, object] | None]:
        """Read one legacy JSON file and classify it as ok, missing, or invalid."""

        try:
            stat_result = path.lstat()
        except FileNotFoundError:
            return "missing", None
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to inspect legacy integration store file: {path}") from exc

        if path.is_symlink():
            raise IntegrationStoreError(f"Refusing to use symlinked legacy integration store file: {path}")
        if not path.is_file():
            return "invalid", None
        if stat_result.st_size > _MAX_STORE_BYTES:
            raise IntegrationStoreError(
                f"Legacy integration store exceeds the {_MAX_STORE_BYTES}-byte safety limit: {path}"
            )

        open_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW

        try:
            descriptor = os.open(path, open_flags)
            with os.fdopen(descriptor, "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except FileNotFoundError:
            return "missing", None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "invalid", None
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to read legacy integration store file: {path}") from exc

        if not isinstance(payload, dict):
            return "invalid", None

        payload.setdefault("version", 1)
        integrations = payload.setdefault("integrations", {})
        if not isinstance(integrations, dict):
            return "invalid", None
        return "ok", payload

    @contextmanager
    def _bootstrap_lock(self, create_parent: bool) -> Iterator[None]:
        """Hold thread and process locks around one migration or recovery critical section."""

        thread_lock = self._get_thread_lock()
        with thread_lock:
            if create_parent:
                self._ensure_store_parent()
            else:
                self._ensure_no_symlink_ancestors(self.path.parent)
                if not self.path.parent.exists():
                    yield
                    return

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
        """Create and validate the final store directory."""

        self._ensure_no_symlink_ancestors(self.path.parent)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # BREAKING: tighten the final store directory to 0700 so SQLite sidecars remain private.
            os.chmod(self.path.parent, _SECURE_DIR_MODE)
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
        """Open the bootstrap lock file used for cross-process migration and recovery."""

        try:
            if self._bootstrap_lock_path.is_symlink():
                raise IntegrationStoreError(
                    f"Refusing symlinked integration store lock file: {self._bootstrap_lock_path}"
                )
        except OSError as exc:
            raise IntegrationStoreError(
                f"Failed to inspect integration store lock file: {self._bootstrap_lock_path}"
            ) from exc

        open_flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW

        try:
            descriptor = os.open(self._bootstrap_lock_path, open_flags, _SECURE_FILE_MODE)
            os.chmod(self._bootstrap_lock_path, _SECURE_FILE_MODE)
        except OSError as exc:
            raise IntegrationStoreError(
                f"Failed to open integration store lock file: {self._bootstrap_lock_path}"
            ) from exc
        return os.fdopen(descriptor, "a+b", buffering=0)

    def _replace_database_file(self, source_path: Path, destination_path: Path) -> None:
        """Atomically replace one SQLite database file and clear stale sidecars."""

        try:
            os.chmod(source_path, _SECURE_FILE_MODE)
            os.replace(source_path, destination_path)
            os.chmod(destination_path, _SECURE_FILE_MODE)
            self._remove_sqlite_sidecars(destination_path)
            self._fsync_directory(destination_path.parent)
        except OSError as exc:
            raise IntegrationStoreError(f"Failed to replace SQLite integration store file: {destination_path}") from exc

    def _remove_sqlite_sidecars(self, base_path: Path) -> None:
        """Delete stale -wal and -shm files belonging to one database file."""

        for suffix in ("-wal", "-shm"):
            candidate = base_path.with_name(f"{base_path.name}{suffix}")
            try:
                os.unlink(candidate)
            except FileNotFoundError:
                pass
            except OSError:
                pass

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
