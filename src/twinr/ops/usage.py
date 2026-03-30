# CHANGELOG: 2026-03-30
# BUG-1: Fixes silent token/model undercounting by normalizing 2026 OpenAI / Anthropic / Gemini usage schemas.
# BUG-2: Replaces O(n) JSONL rescans with indexed SQLite queries so tail()/summary() remain fast on Pi 4 deployments.
# SEC-1: Adds recursive, size-bounded metadata redaction so nested headers / prompt-like fields do not hit disk verbatim.
# SEC-2: Adds bounded retention + compaction to prevent practical disk-exhaustion / SD-card wear-out from unbounded telemetry growth.
# IMP-1: Upgrades the primary store from ad-hoc JSONL to a hardened SQLite sidecar with schema versioning and legacy JSONL import.
# IMP-2: Tracks provider, cache-creation tokens, cache-read tokens, reasoning tokens, audio tokens, and tool-prompt tokens.

"""Persist normalized model-usage telemetry for Twinr runtime paths.

# BREAKING:
# If `path` points at a legacy JSONL file such as `usage.jsonl`, the primary
# store now lives in a SQLite sidecar: `usage.jsonl.sqlite3`. Historical JSONL
# rows are imported automatically.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import stat
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.ops.paths import resolve_ops_paths, resolve_ops_paths_for_config


LOGGER = logging.getLogger(__name__)

_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_DB_USER_VERSION = 2

_DEFAULT_BUSY_TIMEOUT_MS = 5_000
_DEFAULT_MAX_RECORDS = 50_000
_DEFAULT_RETENTION_HOURS = 24 * 30
_DEFAULT_WAL_AUTOCHECKPOINT_PAGES = 256
_DEFAULT_PRUNE_INTERVAL = 64

_ENV_BUSY_TIMEOUT_MS = "TWINR_USAGE_BUSY_TIMEOUT_MS"
_ENV_MAX_RECORDS = "TWINR_USAGE_MAX_RECORDS"
_ENV_RETENTION_HOURS = "TWINR_USAGE_RETENTION_HOURS"
_ENV_JOURNAL_MODE = "TWINR_USAGE_SQLITE_JOURNAL_MODE"

_DB_SUFFIXES = {".db", ".sqlite", ".sqlite3"}

_SENSITIVE_METADATA_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "password",
    "secret",
    "session",
    "token",
    "credential",
    "credentials",
)
_CONTENT_METADATA_KEY_PARTS = (
    "prompt",
    "input",
    "output",
    "message",
    "messages",
    "instruction",
    "instructions",
    "content",
    "transcript",
    "request_body",
    "response_body",
    "tool_args",
    "tool_arguments",
    "tool_output",
    "tool_result",
)
_REDACTABLE_CONTAINER_KEY_PARTS = ("header", "headers", "env", "environment", "config")

_METADATA_MAX_ITEMS = 32
_METADATA_MAX_KEY_CHARS = 96
_METADATA_MAX_VALUE_CHARS = 512
_METADATA_MAX_TOTAL_CHARS = 4_096
_FIELD_MAX_CHARS = 256


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso_z() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_str(value: object) -> str:
    try:
        return str(value)
    except Exception:  # pragma: no cover
        return repr(type(value))


def _stable_json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    digest = _digest_text(value)[:12]
    keep = max(16, max_chars - 24)
    return f"{value[:keep]}…[{digest}]"


def _normalize_optional_text(
    value: object,
    *,
    default: str | None = None,
    max_chars: int | None = None,
) -> str | None:
    text = _safe_str(value).strip() if value is not None else ""
    if not text:
        return default
    if max_chars is not None:
        text = _truncate_text(text, max_chars=max_chars)
    return text


def _field_value(source: object, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(key)
    try:
        return getattr(source, key)
    except AttributeError:
        return None
    except Exception:
        LOGGER.debug("Failed to read %r from %r", key, type(source), exc_info=True)
        return None


def _field_value_any(source: object, *keys: str) -> object | None:
    for key in keys:
        value = _field_value(source, key)
        if value is not None:
            return value
    return None


def _first_non_none(*values: object) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _epoch_seconds(value: datetime | None, *, fallback: int = 0) -> int:
    return fallback if value is None else int(value.timestamp())


def _read_env_positive_int(name: str, default: int) -> int:
    value = _coerce_int(os.getenv(name))
    return default if value is None or value <= 0 else value


def _is_sensitive_metadata_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS)


def _is_content_metadata_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in _CONTENT_METADATA_KEY_PARTS)


def _is_redactable_container_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(part in normalized for part in _REDACTABLE_CONTAINER_KEY_PARTS)


def _looks_sensitive_value(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return False
    markers = (
        "authorization:",
        "bearer ",
        "x-api-key",
        "api_key=",
        "apikey=",
        "cookie:",
        "session=",
        "password=",
        "secret=",
        "token=",
        "sk-",
    )
    return any(marker in normalized for marker in markers)


def _sanitize_metadata_value(value: object, *, parent_key: str | None = None) -> str | None:
    if value is None:
        return None

    if parent_key and (
        _is_sensitive_metadata_key(parent_key)
        or _is_content_metadata_key(parent_key)
        or _is_redactable_container_key(parent_key)
    ):
        return "[REDACTED]"

    if isinstance(value, Mapping):
        sanitized: dict[str, str] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= _METADATA_MAX_ITEMS:
                sanitized["__truncated_items__"] = "[TRUNCATED]"
                break
            key = _normalize_optional_text(raw_key, max_chars=_METADATA_MAX_KEY_CHARS)
            if not key:
                continue
            item = _sanitize_metadata_value(raw_value, parent_key=key)
            if item:
                sanitized[key] = item
        return None if not sanitized else _truncate_text(_stable_json_dumps(sanitized), max_chars=_METADATA_MAX_VALUE_CHARS)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        items: list[str] = []
        for index, item in enumerate(value):
            if index >= _METADATA_MAX_ITEMS:
                items.append("[TRUNCATED]")
                break
            rendered = _sanitize_metadata_value(item, parent_key=parent_key)
            if rendered:
                items.append(rendered)
        return None if not items else _truncate_text(_stable_json_dumps(items), max_chars=_METADATA_MAX_VALUE_CHARS)

    if isinstance(value, (bytes, bytearray, memoryview)):
        rendered = f"[{type(value).__name__}:{len(value)} bytes]"
    else:
        rendered = _safe_str(value).strip()
    if not rendered:
        return None
    if _looks_sensitive_value(rendered):
        return "[REDACTED]"
    return _truncate_text(rendered, max_chars=_METADATA_MAX_VALUE_CHARS)


def _sanitize_metadata(metadata: object | None) -> dict[str, str] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        LOGGER.debug("Ignoring non-mapping metadata of type %r", type(metadata))
        return None

    sanitized: dict[str, str] = {}
    total_chars = 0
    for index, (raw_key, raw_value) in enumerate(metadata.items()):
        if index >= _METADATA_MAX_ITEMS:
            sanitized["__truncated_items__"] = "[TRUNCATED]"
            break
        key = _normalize_optional_text(raw_key, max_chars=_METADATA_MAX_KEY_CHARS)
        if not key:
            continue
        value = _sanitize_metadata_value(raw_value, parent_key=key)
        if not value:
            continue
        projected = total_chars + len(key) + len(value)
        if projected > _METADATA_MAX_TOTAL_CHARS:
            sanitized["__truncated_total__"] = "[TRUNCATED]"
            break
        sanitized[key] = value
        total_chars = projected
    return sanitized or None


def _assert_no_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            if component.is_symlink():
                raise ValueError(f"Unsafe usage store path component: {component}")
        except OSError as exc:
            raise ValueError(f"Unable to validate usage store path {path}: {exc}") from exc


def _assert_regular_file_target(path: Path) -> None:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"Unable to stat usage store path {path}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Usage store path must be a regular file: {path}")


def _chmod_if_possible(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        LOGGER.debug("Failed to chmod %s to %#o", path, mode, exc_info=True)


def _normalize_store_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    _assert_no_symlink_components(candidate)
    normalized = Path(os.path.normpath(os.fspath(candidate)))
    _assert_no_symlink_components(normalized)
    _assert_regular_file_target(normalized)
    return normalized


def _ensure_parent_dir(path: Path) -> None:
    _assert_no_symlink_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
    _chmod_if_possible(path.parent, _PRIVATE_DIR_MODE)
    _assert_no_symlink_components(path.parent)
    _assert_regular_file_target(path)


def _derive_db_path(base_path: Path) -> Path:
    if base_path.suffix.lower() in _DB_SUFFIXES:
        return base_path
    return _normalize_store_path(f"{base_path}.sqlite3")


def _derive_sidecar_path(path: Path, suffix: str) -> Path:
    return _normalize_store_path(f"{path}{suffix}")


@contextmanager
def _locked_sidecar(path: Path):
    _ensure_parent_dir(path)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, _PRIVATE_FILE_MODE)
    try:
        try:
            os.fchmod(fd, _PRIVATE_FILE_MODE)
        except OSError:
            LOGGER.debug("Failed to tighten permissions on %s", path, exc_info=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        os.close(fd)


def _open_locked_read_text(path: Path):
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        return os.fdopen(fd, "r", encoding="utf-8", errors="replace", closefd=True)
    except Exception:
        os.close(fd)
        raise


def _sqlite_version_info() -> tuple[int, int, int]:
    version = getattr(sqlite3, "sqlite_version_info", None)
    if isinstance(version, tuple) and len(version) >= 3:
        return int(version[0]), int(version[1]), int(version[2])
    parts = (getattr(sqlite3, "sqlite_version", "0.0.0").split(".") + ["0", "0", "0"])[:3]
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return 0, 0, 0


def _sqlite_has_wal_reset_fix(version: tuple[int, int, int] | None = None) -> bool:
    major, minor, patch = version or _sqlite_version_info()
    if major != 3:
        return False
    if minor >= 52:
        return True
    if minor == 51 and patch >= 3:
        return True
    if minor == 50 and patch >= 7:
        return True
    if minor == 44 and patch >= 6:
        return True
    return False


def _select_journal_mode() -> str:
    override = (os.getenv(_ENV_JOURNAL_MODE) or "").strip().upper()
    if override in {"DELETE", "WAL"}:
        return override
    return "WAL" if _sqlite_has_wal_reset_fix() else "DELETE"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    audio_input_tokens: int | None = None
    audio_output_tokens: int | None = None
    tool_input_tokens: int | None = None

    @property
    def has_values(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.cached_input_tokens,
                self.cache_creation_input_tokens,
                self.reasoning_tokens,
                self.audio_input_tokens,
                self.audio_output_tokens,
                self.tool_input_tokens,
            )
        )

    @property
    def total_tokens_estimate(self) -> int | None:
        if self.total_tokens is not None:
            return self.total_tokens
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return int(self.input_tokens or 0) + int(self.output_tokens or 0)

    def to_dict(self) -> dict[str, int]:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        if "total_tokens" not in payload and self.total_tokens_estimate is not None:
            payload["total_tokens"] = self.total_tokens_estimate
        return payload


@dataclass(frozen=True, slots=True)
class UsageRecord:
    created_at: str
    source: str
    request_kind: str
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    request_id: str | None = None
    used_web_search: bool | None = None
    token_usage: TokenUsage | None = None
    metadata: dict[str, str] | None = None

    @property
    def total_tokens(self) -> int | None:
        return None if self.token_usage is None else self.token_usage.total_tokens_estimate

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "created_at": self.created_at,
            "source": self.source,
            "request_kind": self.request_kind,
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "used_web_search": self.used_web_search,
            "metadata": dict(self.metadata or {}),
        }
        if self.token_usage is not None and self.token_usage.has_values:
            payload["token_usage"] = self.token_usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "UsageRecord":
        raw_usage = payload.get("token_usage")
        token_usage = None
        if isinstance(raw_usage, Mapping):
            token_usage = TokenUsage(
                input_tokens=_coerce_int(raw_usage.get("input_tokens")),
                output_tokens=_coerce_int(raw_usage.get("output_tokens")),
                total_tokens=_coerce_int(raw_usage.get("total_tokens")),
                cached_input_tokens=_coerce_int(raw_usage.get("cached_input_tokens")),
                cache_creation_input_tokens=_coerce_int(raw_usage.get("cache_creation_input_tokens")),
                reasoning_tokens=_coerce_int(raw_usage.get("reasoning_tokens")),
                audio_input_tokens=_coerce_int(raw_usage.get("audio_input_tokens")),
                audio_output_tokens=_coerce_int(raw_usage.get("audio_output_tokens")),
                tool_input_tokens=_coerce_int(raw_usage.get("tool_input_tokens")),
            )
        return cls(
            created_at=_normalize_optional_text(payload.get("created_at")) or "",
            source=_normalize_optional_text(payload.get("source"), default="runtime", max_chars=_FIELD_MAX_CHARS) or "runtime",
            request_kind=_normalize_optional_text(payload.get("request_kind"), default="unknown", max_chars=_FIELD_MAX_CHARS) or "unknown",
            provider=_normalize_optional_text(payload.get("provider"), max_chars=_FIELD_MAX_CHARS),
            model=_normalize_optional_text(payload.get("model"), max_chars=_FIELD_MAX_CHARS),
            response_id=_normalize_optional_text(payload.get("response_id"), max_chars=_FIELD_MAX_CHARS),
            request_id=_normalize_optional_text(payload.get("request_id"), max_chars=_FIELD_MAX_CHARS),
            used_web_search=_coerce_bool(payload.get("used_web_search")),
            token_usage=token_usage if token_usage is not None and token_usage.has_values else None,
            metadata=_sanitize_metadata(payload.get("metadata")),
        )


@dataclass(frozen=True, slots=True)
class UsageSummary:
    requests_total: int = 0
    requests_with_token_data: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0
    tool_input_tokens: int = 0
    latest_provider: str | None = None
    latest_model: str | None = None
    latest_request_kind: str | None = None
    latest_created_at: str | None = None
    by_kind: dict[str, int] | None = None
    by_model: dict[str, int] | None = None
    by_provider: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requests_total": self.requests_total,
            "requests_with_token_data": self.requests_with_token_data,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_input_tokens": self.audio_input_tokens,
            "audio_output_tokens": self.audio_output_tokens,
            "tool_input_tokens": self.tool_input_tokens,
            "latest_provider": self.latest_provider,
            "latest_model": self.latest_model,
            "latest_request_kind": self.latest_request_kind,
            "latest_created_at": self.latest_created_at,
            "by_kind": dict(self.by_kind or {}),
            "by_model": dict(self.by_model or {}),
            "by_provider": dict(self.by_provider or {}),
        }


class TwinrUsageStore:
    """Persist and summarize Twinr usage telemetry."""

    def __init__(self, path: str | Path) -> None:
        base_path = _normalize_store_path(path)
        self.path = base_path
        self.db_path = _derive_db_path(base_path)
        self.legacy_path = None if self.db_path == base_path else base_path
        self.lock_path = _derive_sidecar_path(self.db_path, ".lock")
        self.busy_timeout_ms = _read_env_positive_int(_ENV_BUSY_TIMEOUT_MS, _DEFAULT_BUSY_TIMEOUT_MS)
        self.max_records = _read_env_positive_int(_ENV_MAX_RECORDS, _DEFAULT_MAX_RECORDS)
        self.retention_hours = _read_env_positive_int(_ENV_RETENTION_HOURS, _DEFAULT_RETENTION_HOURS)
        self.journal_mode = _select_journal_mode()
        self._init_lock = threading.Lock()
        self._initialized = False

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TwinrUsageStore":
        return cls(resolve_ops_paths_for_config(config).usage_path)

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "TwinrUsageStore":
        return cls(resolve_ops_paths(project_root).usage_path)

    def _connect(self) -> sqlite3.Connection:
        _ensure_parent_dir(self.db_path)
        connection = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _configure_connection(self, connection: sqlite3.Connection) -> None:
        actual = connection.execute(f"PRAGMA journal_mode={self.journal_mode}").fetchone()[0]
        actual_mode = str(actual).upper()
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        if actual_mode == "WAL":
            connection.execute(f"PRAGMA wal_autocheckpoint={_DEFAULT_WAL_AUTOCHECKPOINT_PAGES}")

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                source TEXT NOT NULL,
                request_kind TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                response_id TEXT,
                request_id TEXT,
                used_web_search INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cached_input_tokens INTEGER,
                cache_creation_input_tokens INTEGER,
                reasoning_tokens INTEGER,
                audio_input_tokens INTEGER,
                audio_output_tokens INTEGER,
                tool_input_tokens INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                record_digest TEXT NOT NULL UNIQUE
            ) STRICT
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) STRICT
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_records(created_at_epoch DESC, id DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_usage_kind_created ON usage_records(request_kind, created_at_epoch DESC)")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_model_created
            ON usage_records(model, created_at_epoch DESC)
            WHERE model IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_usage_provider_created
            ON usage_records(provider, created_at_epoch DESC)
            WHERE provider IS NOT NULL
            """
        )
        connection.execute(f"PRAGMA user_version={_DB_USER_VERSION}")
        _chmod_if_possible(self.db_path, _PRIVATE_FILE_MODE)

    def _meta_get(self, connection: sqlite3.Connection, key: str) -> str | None:
        row = connection.execute("SELECT value FROM usage_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else _normalize_optional_text(row["value"])

    def _meta_set(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            """
            INSERT INTO usage_meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _file_signature(self, path: Path) -> str | None:
        try:
            st = path.stat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            LOGGER.debug("Failed to stat %s: %s", path, exc, exc_info=True)
            return None
        return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"

    def _iter_jsonl_records(self, path: Path | None) -> Iterator[UsageRecord]:
        if path is None or not path.exists():
            return
        try:
            with _open_locked_read_text(path) as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        LOGGER.warning("Skipping invalid usage JSON line %s in %s", line_number, path)
                        continue
                    if isinstance(parsed, Mapping):
                        yield UsageRecord.from_dict(parsed)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            LOGGER.warning("Failed to read legacy usage records from %s: %s", path, exc)
            return

    def _record_epoch(self, record: UsageRecord) -> int:
        return _epoch_seconds(_parse_iso_datetime(record.created_at), fallback=0)

    def _record_digest(self, record: UsageRecord) -> str:
        return _digest_text(_stable_json_dumps(record.to_dict()))

    def _insert_record(self, connection: sqlite3.Connection, record: UsageRecord) -> None:
        usage = record.token_usage if record.token_usage is not None and record.token_usage.has_values else None
        connection.execute(
            """
            INSERT OR IGNORE INTO usage_records (
                created_at,
                created_at_epoch,
                source,
                request_kind,
                provider,
                model,
                response_id,
                request_id,
                used_web_search,
                input_tokens,
                output_tokens,
                total_tokens,
                cached_input_tokens,
                cache_creation_input_tokens,
                reasoning_tokens,
                audio_input_tokens,
                audio_output_tokens,
                tool_input_tokens,
                metadata_json,
                record_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.created_at,
                self._record_epoch(record),
                record.source,
                record.request_kind,
                record.provider,
                record.model,
                record.response_id,
                record.request_id,
                None if record.used_web_search is None else int(record.used_web_search),
                None if usage is None else usage.input_tokens,
                None if usage is None else usage.output_tokens,
                None if usage is None else usage.total_tokens_estimate,
                None if usage is None else usage.cached_input_tokens,
                None if usage is None else usage.cache_creation_input_tokens,
                None if usage is None else usage.reasoning_tokens,
                None if usage is None else usage.audio_input_tokens,
                None if usage is None else usage.audio_output_tokens,
                None if usage is None else usage.tool_input_tokens,
                _stable_json_dumps(record.metadata or {}),
                self._record_digest(record),
            ),
        )

    def _sync_legacy_jsonl(self, connection: sqlite3.Connection) -> None:
        signature = self._file_signature(self.legacy_path) if self.legacy_path is not None else None
        if not signature:
            return
        if self._meta_get(connection, "legacy_jsonl_signature") == signature:
            return
        for record in self._iter_jsonl_records(self.legacy_path):
            self._insert_record(connection, record)
        self._meta_set(connection, "legacy_jsonl_signature", signature)

    def _prune_if_needed(self, connection: sqlite3.Connection, *, force: bool = False) -> bool:
        if self.max_records <= 0 and self.retention_hours <= 0:
            return False
        max_id_row = connection.execute("SELECT MAX(id) AS max_id FROM usage_records").fetchone()
        max_id = int(max_id_row["max_id"] or 0)
        if not force and (max_id == 0 or max_id % _DEFAULT_PRUNE_INTERVAL != 0):
            return False

        if self.retention_hours > 0:
            cutoff_epoch = _epoch_seconds(_utc_now() - timedelta(hours=self.retention_hours))
            connection.execute(
                "DELETE FROM usage_records WHERE created_at_epoch > 0 AND created_at_epoch < ?",
                (cutoff_epoch,),
            )

        if self.max_records > 0:
            connection.execute(
                """
                DELETE FROM usage_records
                WHERE id IN (
                    SELECT id
                    FROM usage_records
                    ORDER BY created_at_epoch DESC, id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.max_records,),
            )

        return True

    def _post_commit_housekeeping(self, connection: sqlite3.Connection) -> None:
        try:
            if self.journal_mode == "WAL":
                connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
            else:
                connection.execute("PRAGMA incremental_vacuum(64)")
        except sqlite3.DatabaseError:
            LOGGER.debug("Post-commit housekeeping failed", exc_info=True)

    def _ensure_initialized(self) -> None:
        with self._init_lock:
            with _locked_sidecar(self.lock_path):
                connection = self._connect()
                try:
                    self._configure_connection(connection)
                    self._create_schema(connection)
                    connection.execute("BEGIN IMMEDIATE")
                    self._sync_legacy_jsonl(connection)
                    pruned = self._prune_if_needed(connection, force=not self._initialized)
                    connection.execute("COMMIT")
                    if pruned:
                        self._post_commit_housekeeping(connection)
                    self._initialized = True
                except Exception:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.DatabaseError:
                        pass
                    raise
                finally:
                    connection.close()

    def append(
        self,
        *,
        source: str,
        request_kind: str,
        provider: str | None = None,
        model: str | None = None,
        response_id: str | None = None,
        request_id: str | None = None,
        used_web_search: bool | None = None,
        token_usage: TokenUsage | None = None,
        metadata: dict[str, object] | None = None,
    ) -> UsageRecord:
        """Append one normalized usage record and return the normalized entry."""

        record = UsageRecord(
            created_at=_utc_now_iso_z(),
            source=_normalize_optional_text(source, default="runtime", max_chars=_FIELD_MAX_CHARS) or "runtime",
            request_kind=_normalize_optional_text(request_kind, default="unknown", max_chars=_FIELD_MAX_CHARS) or "unknown",
            provider=_normalize_optional_text(provider, max_chars=_FIELD_MAX_CHARS),
            model=_normalize_optional_text(model, max_chars=_FIELD_MAX_CHARS),
            response_id=_normalize_optional_text(response_id, max_chars=_FIELD_MAX_CHARS),
            request_id=_normalize_optional_text(request_id, max_chars=_FIELD_MAX_CHARS),
            used_web_search=_coerce_bool(used_web_search),
            token_usage=token_usage if token_usage is not None and token_usage.has_values else None,
            metadata=_sanitize_metadata(metadata),
        )

        try:
            self._ensure_initialized()
            with _locked_sidecar(self.lock_path):
                connection = self._connect()
                try:
                    self._configure_connection(connection)
                    connection.execute("BEGIN IMMEDIATE")
                    self._insert_record(connection, record)
                    pruned = self._prune_if_needed(connection)
                    connection.execute("COMMIT")
                    if pruned:
                        self._post_commit_housekeeping(connection)
                except Exception:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.DatabaseError:
                        pass
                    raise
                finally:
                    connection.close()
        except Exception as exc:
            LOGGER.warning("Failed to append usage record to %s: %s", self.db_path, exc)

        return record

    def _row_to_record(self, row: sqlite3.Row) -> UsageRecord:
        metadata: dict[str, str] | None = None
        raw_metadata = row["metadata_json"]
        if raw_metadata:
            try:
                metadata = _sanitize_metadata(json.loads(raw_metadata))
            except json.JSONDecodeError:
                metadata = None

        usage = TokenUsage(
            input_tokens=_coerce_int(row["input_tokens"]),
            output_tokens=_coerce_int(row["output_tokens"]),
            total_tokens=_coerce_int(row["total_tokens"]),
            cached_input_tokens=_coerce_int(row["cached_input_tokens"]),
            cache_creation_input_tokens=_coerce_int(row["cache_creation_input_tokens"]),
            reasoning_tokens=_coerce_int(row["reasoning_tokens"]),
            audio_input_tokens=_coerce_int(row["audio_input_tokens"]),
            audio_output_tokens=_coerce_int(row["audio_output_tokens"]),
            tool_input_tokens=_coerce_int(row["tool_input_tokens"]),
        )

        return UsageRecord(
            created_at=_normalize_optional_text(row["created_at"]) or "",
            source=_normalize_optional_text(row["source"], default="runtime") or "runtime",
            request_kind=_normalize_optional_text(row["request_kind"], default="unknown") or "unknown",
            provider=_normalize_optional_text(row["provider"]),
            model=_normalize_optional_text(row["model"]),
            response_id=_normalize_optional_text(row["response_id"]),
            request_id=_normalize_optional_text(row["request_id"]),
            used_web_search=None if row["used_web_search"] is None else bool(int(row["used_web_search"])),
            token_usage=usage if usage.has_values else None,
            metadata=metadata,
        )

    def tail(self, *, limit: int = 100) -> list[UsageRecord]:
        """Return the most recent usage records in chronological order."""

        if limit <= 0:
            return []
        try:
            self._ensure_initialized()
            connection = self._connect()
            try:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM usage_records
                    ORDER BY created_at_epoch DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [self._row_to_record(row) for row in reversed(rows)]
            finally:
                connection.close()
        except Exception as exc:
            LOGGER.warning("Failed to read usage tail from %s: %s", self.db_path, exc)
            return list(deque(self._iter_jsonl_records(self.legacy_path), maxlen=limit))

    def summary(self, *, within_hours: int | None = None) -> UsageSummary:
        """Summarize stored usage records for the optional recent window."""

        cutoff_epoch: int | None = None
        if within_hours is not None and within_hours > 0:
            cutoff_epoch = _epoch_seconds(_utc_now() - timedelta(hours=within_hours))

        try:
            self._ensure_initialized()
            connection = self._connect()
            try:
                where_sql = ""
                params: tuple[object, ...] = ()
                if cutoff_epoch is not None:
                    where_sql = "WHERE created_at_epoch >= ?"
                    params = (cutoff_epoch,)

                aggregate = connection.execute(
                    f"""
                    SELECT
                        COUNT(*) AS requests_total,
                        SUM(
                            CASE
                                WHEN input_tokens IS NOT NULL
                                  OR output_tokens IS NOT NULL
                                  OR total_tokens IS NOT NULL
                                  OR cached_input_tokens IS NOT NULL
                                  OR cache_creation_input_tokens IS NOT NULL
                                  OR reasoning_tokens IS NOT NULL
                                  OR audio_input_tokens IS NOT NULL
                                  OR audio_output_tokens IS NOT NULL
                                  OR tool_input_tokens IS NOT NULL
                                THEN 1 ELSE 0
                            END
                        ) AS requests_with_token_data,
                        COALESCE(SUM(COALESCE(input_tokens, 0)), 0) AS total_input_tokens,
                        COALESCE(SUM(COALESCE(output_tokens, 0)), 0) AS total_output_tokens,
                        COALESCE(SUM(COALESCE(total_tokens, 0)), 0) AS total_tokens,
                        COALESCE(SUM(COALESCE(cached_input_tokens, 0)), 0) AS cached_input_tokens,
                        COALESCE(SUM(COALESCE(cache_creation_input_tokens, 0)), 0) AS cache_creation_input_tokens,
                        COALESCE(SUM(COALESCE(reasoning_tokens, 0)), 0) AS reasoning_tokens,
                        COALESCE(SUM(COALESCE(audio_input_tokens, 0)), 0) AS audio_input_tokens,
                        COALESCE(SUM(COALESCE(audio_output_tokens, 0)), 0) AS audio_output_tokens,
                        COALESCE(SUM(COALESCE(tool_input_tokens, 0)), 0) AS tool_input_tokens
                    FROM usage_records
                    {where_sql}
                    """,
                    params,
                ).fetchone()

                requests_total = int(aggregate["requests_total"] or 0)
                if requests_total == 0:
                    return UsageSummary(by_kind={}, by_model={}, by_provider={})

                latest = connection.execute(
                    f"""
                    SELECT created_at, provider, model, request_kind
                    FROM usage_records
                    {where_sql}
                    ORDER BY created_at_epoch DESC, id DESC
                    LIMIT 1
                    """,
                    params,
                ).fetchone()

                by_kind = {
                    str(row["request_kind"]): int(row["count"])
                    for row in connection.execute(
                        f"""
                        SELECT request_kind, COUNT(*) AS count
                        FROM usage_records
                        {where_sql}
                        GROUP BY request_kind
                        ORDER BY count DESC, request_kind ASC
                        """,
                        params,
                    )
                }
                by_model = {
                    str(row["model_key"]): int(row["count"])
                    for row in connection.execute(
                        f"""
                        SELECT COALESCE(NULLIF(model, ''), 'unknown') AS model_key, COUNT(*) AS count
                        FROM usage_records
                        {where_sql}
                        GROUP BY model_key
                        ORDER BY count DESC, model_key ASC
                        """,
                        params,
                    )
                }
                by_provider = {
                    str(row["provider_key"]): int(row["count"])
                    for row in connection.execute(
                        f"""
                        SELECT COALESCE(NULLIF(provider, ''), 'unknown') AS provider_key, COUNT(*) AS count
                        FROM usage_records
                        {where_sql}
                        GROUP BY provider_key
                        ORDER BY count DESC, provider_key ASC
                        """,
                        params,
                    )
                }

                return UsageSummary(
                    requests_total=requests_total,
                    requests_with_token_data=int(aggregate["requests_with_token_data"] or 0),
                    total_input_tokens=int(aggregate["total_input_tokens"] or 0),
                    total_output_tokens=int(aggregate["total_output_tokens"] or 0),
                    total_tokens=int(aggregate["total_tokens"] or 0),
                    cached_input_tokens=int(aggregate["cached_input_tokens"] or 0),
                    cache_creation_input_tokens=int(aggregate["cache_creation_input_tokens"] or 0),
                    reasoning_tokens=int(aggregate["reasoning_tokens"] or 0),
                    audio_input_tokens=int(aggregate["audio_input_tokens"] or 0),
                    audio_output_tokens=int(aggregate["audio_output_tokens"] or 0),
                    tool_input_tokens=int(aggregate["tool_input_tokens"] or 0),
                    latest_provider=None if latest is None else _normalize_optional_text(latest["provider"]),
                    latest_model=None if latest is None else _normalize_optional_text(latest["model"]),
                    latest_request_kind=None if latest is None else _normalize_optional_text(latest["request_kind"]),
                    latest_created_at=None if latest is None else _normalize_optional_text(latest["created_at"]),
                    by_kind=by_kind,
                    by_model=by_model,
                    by_provider=by_provider,
                )
            finally:
                connection.close()
        except Exception as exc:
            LOGGER.warning("Failed to summarize usage records from %s: %s", self.db_path, exc)
            records = list(self._iter_jsonl_records(self.legacy_path))
            if cutoff_epoch is not None:
                records = [r for r in records if self._record_epoch(r) >= cutoff_epoch]
            return _summary_from_records(records)


def _summary_from_records(records: Sequence[UsageRecord]) -> UsageSummary:
    if not records:
        return UsageSummary(by_kind={}, by_model={}, by_provider={})

    by_kind: dict[str, int] = {}
    by_model: dict[str, int] = {}
    by_provider: dict[str, int] = {}
    latest: UsageRecord | None = None
    latest_dt: datetime | None = None

    requests_with_token_data = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    cached_input_tokens = 0
    cache_creation_input_tokens = 0
    reasoning_tokens = 0
    audio_input_tokens = 0
    audio_output_tokens = 0
    tool_input_tokens = 0

    for record in records:
        by_kind[record.request_kind] = by_kind.get(record.request_kind, 0) + 1
        by_model[record.model or "unknown"] = by_model.get(record.model or "unknown", 0) + 1
        by_provider[record.provider or "unknown"] = by_provider.get(record.provider or "unknown", 0) + 1

        dt = _parse_iso_datetime(record.created_at)
        if latest is None or (dt is not None and (latest_dt is None or dt >= latest_dt)):
            latest = record
            latest_dt = dt

        if record.token_usage is None:
            continue
        requests_with_token_data += 1
        total_input_tokens += int(record.token_usage.input_tokens or 0)
        total_output_tokens += int(record.token_usage.output_tokens or 0)
        total_tokens += int(record.token_usage.total_tokens_estimate or 0)
        cached_input_tokens += int(record.token_usage.cached_input_tokens or 0)
        cache_creation_input_tokens += int(record.token_usage.cache_creation_input_tokens or 0)
        reasoning_tokens += int(record.token_usage.reasoning_tokens or 0)
        audio_input_tokens += int(record.token_usage.audio_input_tokens or 0)
        audio_output_tokens += int(record.token_usage.audio_output_tokens or 0)
        tool_input_tokens += int(record.token_usage.tool_input_tokens or 0)

    return UsageSummary(
        requests_total=len(records),
        requests_with_token_data=requests_with_token_data,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        reasoning_tokens=reasoning_tokens,
        audio_input_tokens=audio_input_tokens,
        audio_output_tokens=audio_output_tokens,
        tool_input_tokens=tool_input_tokens,
        latest_provider=None if latest is None else latest.provider,
        latest_model=None if latest is None else latest.model,
        latest_request_kind=None if latest is None else latest.request_kind,
        latest_created_at=None if latest is None else latest.created_at,
        by_kind=by_kind,
        by_model=by_model,
        by_provider=by_provider,
    )


def extract_model_name(source: object, fallback: str | None = None) -> str | None:
    """Extract a normalized model name from a provider object or mapping."""

    value = _field_value_any(source, "model", "model_version", "modelVersion")
    if value is None or not _normalize_optional_text(value):
        return _normalize_optional_text(fallback, max_chars=_FIELD_MAX_CHARS)
    return _normalize_optional_text(value, max_chars=_FIELD_MAX_CHARS)


def extract_token_usage(source: object) -> TokenUsage | None:
    """Extract normalized token usage from a provider object or mapping."""

    usage = _field_value_any(source, "usage", "usage_metadata", "usageMetadata")
    if usage is None and _field_value_any(
        source,
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_token_count",
        "promptTokenCount",
        "candidates_token_count",
        "candidatesTokenCount",
    ) is not None:
        usage = source
    if usage is None:
        return None

    input_details = _field_value_any(usage, "input_tokens_details", "input_token_details", "inputTokensDetails")
    output_details = _field_value_any(usage, "output_tokens_details", "output_token_details", "outputTokensDetails")

    raw_input_tokens = _coerce_int(
        _field_value_any(
            usage,
            "input_tokens",
            "uncached_input_tokens",
            "prompt_token_count",
            "promptTokenCount",
        )
    )
    output_tokens = _coerce_int(
        _field_value_any(
            usage,
            "output_tokens",
            "candidates_token_count",
            "candidatesTokenCount",
        )
    )
    total_tokens = _coerce_int(
        _field_value_any(
            usage,
            "total_tokens",
            "total_token_count",
            "totalTokenCount",
        )
    )

    anthropic_cache_read_tokens = _coerce_int(_field_value_any(usage, "cache_read_input_tokens", "cacheReadInputTokens"))
    anthropic_cache_creation_tokens = _coerce_int(
        _field_value_any(usage, "cache_creation_input_tokens", "cacheCreationInputTokens")
    )
    cached_input_tokens = _coerce_int(
        _first_non_none(
            anthropic_cache_read_tokens,
            _field_value_any(usage, "cached_content_token_count", "cachedContentTokenCount"),
            _field_value_any(input_details, "cached_tokens", "cachedTokens"),
        )
    )
    cache_creation_input_tokens = anthropic_cache_creation_tokens

    if anthropic_cache_read_tokens is not None or anthropic_cache_creation_tokens is not None:
        input_tokens = int(raw_input_tokens or 0) + int(anthropic_cache_read_tokens or 0) + int(anthropic_cache_creation_tokens or 0)
    else:
        input_tokens = raw_input_tokens

    reasoning_tokens = _coerce_int(
        _first_non_none(
            _field_value_any(output_details, "reasoning_tokens", "reasoningTokens"),
            _field_value_any(usage, "thoughts_token_count", "thoughtsTokenCount"),
        )
    )
    audio_input_tokens = _coerce_int(
        _field_value_any(input_details, "audio_tokens", "input_audio_tokens", "inputAudioTokens")
    )
    audio_output_tokens = _coerce_int(
        _field_value_any(output_details, "audio_tokens", "output_audio_tokens", "outputAudioTokens")
    )
    tool_input_tokens = _coerce_int(
        _field_value_any(usage, "tool_use_prompt_token_count", "toolUsePromptTokenCount")
    )

    result = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        reasoning_tokens=reasoning_tokens,
        audio_input_tokens=audio_input_tokens,
        audio_output_tokens=audio_output_tokens,
        tool_input_tokens=tool_input_tokens,
    )
    return result if result.has_values else None