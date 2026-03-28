# CHANGELOG: 2026-03-28
# BUG-1: save() now uses atomic write + file fsync + directory fsync so readers never see half-written JSON and power-loss windows are reduced.
# BUG-2: load() no longer crashes on concurrent delete/replace races around stat/read; malformed or oversized payloads degrade to None without tearing down callers.
# BUG-3: the bounded-cue contract is now enforced on both save and load; non-finite or overly large TTLs are clamped to a safe maximum instead of lingering indefinitely or crashing.
# SEC-1: reads and writes now refuse symlink-following on the final path where supported, blocking practical local symlink overwrite/read attacks in shared artifact directories.
# SEC-2: payload reads are bounded to a small maximum size and restricted to regular files, preventing local DoS via giant files, devices, pipes, or malformed store artifacts.
# IMP-1: persisted payloads now carry optional "_schema_version" metadata for forward migrations.
# BREAKING: strict external JSON parsers that reject unknown top-level keys must allow "_schema_version".
# IMP-2: writer coordination now uses an OS-level lock file on Unix/Pi deployments so concurrent writers serialize cleanly.
# IMP-3: repeated load() calls are stat-cached, reducing JSON parsing and disk churn in polling-heavy HDMI render loops.
# BREAKING: cue expiry is now hard-clamped to max_ttl_s (default 300s) to preserve the "short ambient impulse" contract.

"""Persist short ambient display impulses for Twinr's HDMI reserve area.

The default HDMI scene already has a dedicated right-hand reserve area used by
emoji acknowledgements and richer presentation cards. This module adds a small
text-plus-emoji cue contract for quieter companion impulses that should remain
outside the generic runtime snapshot schema.

The contract is intentionally bounded:

- one active impulse at a time
- short text fields only
- positive visual accents
- explicit expiry so stale companion nudges disappear on their own
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import errno
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import stat
import tempfile
import threading
from typing import Protocol

from twinr.display.emoji_cues import DisplayEmojiSymbol

try:
    import fcntl
except ImportError:  # pragma: no cover - Raspberry Pi / Linux deployments do have fcntl.
    fcntl = None  # type: ignore[assignment]


_DEFAULT_AMBIENT_IMPULSE_TTL_S = 18.0
_DEFAULT_AMBIENT_IMPULSE_MAX_TTL_S = 300.0
_DEFAULT_AMBIENT_IMPULSE_MAX_BYTES = 4_096
_MAX_AMBIENT_IMPULSE_MAX_BYTES = 65_536
_MIN_AMBIENT_IMPULSE_TTL_S = 0.1
_DEFAULT_AMBIENT_IMPULSE_PATH = "artifacts/stores/ops/display_ambient_impulse.json"
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warm"})
_ALLOWED_ACTIONS = frozenset({"hint", "brief_update", "ask_one", "invite_follow_up"})
_SCHEMA_VERSION_KEY = "_schema_version"
_SCHEMA_VERSION = 2
_LOCK_FILE_SUFFIX = ".lock"

_CACHE_STATE_MISSING = "missing"
_CACHE_STATE_INVALID = "invalid"
_CACHE_STATE_VALID = "valid"

_LOGGER = logging.getLogger(__name__)


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the ambient cue store."""

    project_root: str | PathLike[str]


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _normalize_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into an aware UTC datetime."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Normalize one bounded display text field."""

    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_symbol(value: object | None) -> DisplayEmojiSymbol:
    """Normalize one optional symbol token into a supported emoji symbol."""

    if isinstance(value, DisplayEmojiSymbol):
        return value
    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if not compact:
        return DisplayEmojiSymbol.SPARKLES
    try:
        return DisplayEmojiSymbol(compact)
    except ValueError:
        return DisplayEmojiSymbol.SPARKLES


def _normalize_accent(value: object | None) -> str:
    """Normalize one optional visual accent token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACCENTS:
        return "info"
    return compact


def _normalize_action(value: object | None) -> str:
    """Normalize one optional engagement-action token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACTIONS:
        return "hint"
    return compact


def _coerce_finite_float(value: object | None, *, default: float) -> float:
    """Normalize one maybe-float into a finite float fallback."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _coerce_max_ttl_s(value: object | None) -> float:
    """Normalize one configured maximum TTL into a safe positive bound."""

    number = _coerce_finite_float(value, default=_DEFAULT_AMBIENT_IMPULSE_MAX_TTL_S)
    return max(_MIN_AMBIENT_IMPULSE_TTL_S, number)


def _coerce_ttl_s(value: object | None, *, default: float, max_ttl_s: float) -> float:
    """Clamp one TTL to the supported bounded impulse contract."""

    number = _coerce_finite_float(value, default=default)
    return min(max_ttl_s, max(_MIN_AMBIENT_IMPULSE_TTL_S, number))


def _coerce_max_bytes(value: object | None) -> int:
    """Normalize one configured payload-size ceiling into a safe integer."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_AMBIENT_IMPULSE_MAX_BYTES
    return min(_MAX_AMBIENT_IMPULSE_MAX_BYTES, max(512, number))


def _stat_identity(file_stat: os.stat_result) -> tuple[int, int, int, int]:
    """Build one stable-ish cache identity from a stat result."""

    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
    )


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry update to disk when the platform supports it."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True, slots=True)
class DisplayAmbientImpulseCue:
    """Describe one active ambient HDMI reserve-card impulse."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    topic_key: str = ""
    semantic_topic_key: str = ""
    eyebrow: str = ""
    headline: str = ""
    body: str = ""
    symbol: str = DisplayEmojiSymbol.SPARKLES.value
    accent: str = "info"
    action: str = "hint"
    attention_state: str = "background"

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue contract."""

        updated_at = _normalize_timestamp(self.updated_at)
        expires_at = _normalize_timestamp(self.expires_at)
        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "external")
        object.__setattr__(self, "updated_at", _format_timestamp(updated_at) if updated_at is not None else None)
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at) if expires_at is not None else None)
        object.__setattr__(self, "topic_key", _compact_text(self.topic_key, max_len=96).casefold())
        semantic_topic_key = _compact_text(self.semantic_topic_key, max_len=96).casefold()
        object.__setattr__(self, "semantic_topic_key", semantic_topic_key or self.topic_key)
        object.__setattr__(self, "eyebrow", _compact_text(self.eyebrow, max_len=36))
        object.__setattr__(self, "headline", _compact_text(self.headline, max_len=128))
        object.__setattr__(self, "body", _compact_text(self.body, max_len=128))
        object.__setattr__(self, "symbol", _normalize_symbol(self.symbol).value)
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "action", _normalize_action(self.action))
        object.__setattr__(
            self,
            "attention_state",
            _compact_text(self.attention_state, max_len=32).lower() or "background",
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_AMBIENT_IMPULSE_TTL_S,
        max_ttl_s: float = _DEFAULT_AMBIENT_IMPULSE_MAX_TTL_S,
    ) -> "DisplayAmbientImpulseCue":
        """Build one normalized ambient impulse from JSON-style data."""

        safe_max_ttl_s = _coerce_max_ttl_s(max_ttl_s)
        safe_default_ttl_s = _coerce_ttl_s(
            default_ttl_s,
            default=_DEFAULT_AMBIENT_IMPULSE_TTL_S,
            max_ttl_s=safe_max_ttl_s,
        )
        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=safe_default_ttl_s)
        else:
            max_expires_at = updated_at + timedelta(seconds=safe_max_ttl_s)
            if expires_at > max_expires_at:
                expires_at = max_expires_at
        return cls(
            source=_compact_text(payload.get("source"), max_len=80) or "external",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            topic_key=_compact_text(payload.get("topic_key"), max_len=96).casefold(),
            semantic_topic_key=_compact_text(payload.get("semantic_topic_key"), max_len=96).casefold(),
            eyebrow=_compact_text(payload.get("eyebrow"), max_len=36),
            headline=_compact_text(payload.get("headline"), max_len=128),
            body=_compact_text(payload.get("body"), max_len=128),
            symbol=_normalize_symbol(payload.get("symbol")).value,
            accent=_normalize_accent(payload.get("accent")),
            action=_normalize_action(payload.get("action")),
            attention_state=_compact_text(payload.get("attention_state"), max_len=32).lower() or "background",
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping."""

        return asdict(self)

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the cue should still affect the HDMI scene."""

        expires_at = _normalize_timestamp(self.expires_at)
        if expires_at is None:
            return True
        return expires_at >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return one stable render-signature fragment for this cue."""

        return (
            self.source,
            self.updated_at,
            self.expires_at,
            self.topic_key,
            self.semantic_topic_key,
            self.eyebrow,
            self.headline,
            self.body,
            self.symbol,
            self.accent,
            self.action,
            self.attention_state,
        )

    def glyph(self) -> str:
        """Return the rendered Unicode glyph for this cue."""

        return _normalize_symbol(self.symbol).glyph()


@dataclass(slots=True)
class DisplayAmbientImpulseCueStore:
    """Read and write the optional ambient-impulse cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_AMBIENT_IMPULSE_TTL_S
    max_ttl_s: float = _DEFAULT_AMBIENT_IMPULSE_MAX_TTL_S
    max_bytes: int = _DEFAULT_AMBIENT_IMPULSE_MAX_BYTES
    _thread_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _cache_identity: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False)
    _cache_state: str = field(default=_CACHE_STATE_MISSING, init=False, repr=False)
    _cache_cue: DisplayAmbientImpulseCue | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayAmbientImpulseCueStore":
        """Resolve the ambient-impulse cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_ambient_impulse_path", _DEFAULT_AMBIENT_IMPULSE_PATH)
            or _DEFAULT_AMBIENT_IMPULSE_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        configured_max_ttl_s = _coerce_max_ttl_s(
            getattr(config, "display_ambient_impulse_max_ttl_s", _DEFAULT_AMBIENT_IMPULSE_MAX_TTL_S)
        )
        configured_default_ttl_s = _coerce_ttl_s(
            getattr(config, "display_ambient_impulse_ttl_s", _DEFAULT_AMBIENT_IMPULSE_TTL_S),
            default=_DEFAULT_AMBIENT_IMPULSE_TTL_S,
            max_ttl_s=configured_max_ttl_s,
        )
        configured_max_bytes = _coerce_max_bytes(
            getattr(config, "display_ambient_impulse_max_bytes", _DEFAULT_AMBIENT_IMPULSE_MAX_BYTES)
        )
        return cls(
            path=resolved_path,
            default_ttl_s=configured_default_ttl_s,
            max_ttl_s=configured_max_ttl_s,
            max_bytes=configured_max_bytes,
        )

    @property
    def lock_path(self) -> Path:
        """Return the sibling file used for write serialization."""

        return self.path.with_name(f"{self.path.name}{_LOCK_FILE_SUFFIX}")

    def _set_cache_missing(self) -> None:
        """Remember that the store currently has no readable payload."""

        self._cache_identity = None
        self._cache_state = _CACHE_STATE_MISSING
        self._cache_cue = None

    def _set_cache_invalid(self, identity: tuple[int, int, int, int] | None) -> None:
        """Remember one invalid payload identity to avoid repeated work and log spam."""

        self._cache_identity = identity
        self._cache_state = _CACHE_STATE_INVALID
        self._cache_cue = None

    def _set_cache_valid(
        self,
        identity: tuple[int, int, int, int] | None,
        cue: DisplayAmbientImpulseCue,
    ) -> None:
        """Remember one valid parsed payload identity."""

        self._cache_identity = identity
        self._cache_state = _CACHE_STATE_VALID
        self._cache_cue = cue

    def _prepare_parent_dir(self) -> None:
        """Ensure the target directory exists with owner-only defaults for new dirs."""

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    @contextmanager
    def _writer_lock(self):
        """Serialize writers with one OS-level lock file on Unix-like systems."""

        self._prepare_parent_dir()
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            lock_fd = os.open(self.lock_path, flags, 0o600)
        except OSError:
            _LOGGER.warning("Failed to open display ambient impulse lock file %s.", self.lock_path, exc_info=True)
            raise
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)

    def _atomic_write_payload_bytes(self, payload_bytes: bytes) -> None:
        """Durably replace the store file with a fully written temporary file."""

        self._prepare_parent_dir()
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.tmp-",
            dir=str(self.path.parent),
            text=False,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(temp_fd, "wb") as handle:
                handle.write(payload_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
            _fsync_directory(self.path.parent)
        except Exception:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                _LOGGER.warning(
                    "Failed to remove temporary ambient impulse payload %s after write failure.",
                    temp_path,
                    exc_info=True,
                )
            raise

    def _build_store_payload(self, cue: DisplayAmbientImpulseCue) -> dict[str, object]:
        """Build the persisted JSON object with migration metadata."""

        payload = cue.to_dict()
        payload[_SCHEMA_VERSION_KEY] = _SCHEMA_VERSION
        return payload

    def _read_payload_bytes_secure(self) -> tuple[bytes, os.stat_result] | None:
        """Read one bounded payload without following the final symlink component."""

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, "rb") as handle:
            file_stat = os.fstat(handle.fileno())
            raw = handle.read(self.max_bytes + 1)
        return raw, file_stat

    def load(self) -> DisplayAmbientImpulseCue | None:
        """Load the current ambient impulse cue, if one exists and parses."""

        with self._thread_lock:
            try:
                path_stat = os.stat(self.path, follow_symlinks=False)
            except FileNotFoundError:
                self._set_cache_missing()
                return None
            except OSError:
                _LOGGER.warning("Failed to stat display ambient impulse cue at %s.", self.path, exc_info=True)
                self._set_cache_missing()
                return None

            identity = _stat_identity(path_stat)
            if identity == self._cache_identity:
                if self._cache_state == _CACHE_STATE_VALID:
                    return self._cache_cue
                if self._cache_state in {_CACHE_STATE_INVALID, _CACHE_STATE_MISSING}:
                    return None

            if not stat.S_ISREG(path_stat.st_mode):
                _LOGGER.warning(
                    "Ignoring display ambient impulse cue at %s because it is not a regular file.",
                    self.path,
                )
                self._set_cache_invalid(identity)
                return None

            if path_stat.st_size > self.max_bytes:
                _LOGGER.warning(
                    "Ignoring display ambient impulse cue at %s because it exceeds %s bytes.",
                    self.path,
                    self.max_bytes,
                )
                self._set_cache_invalid(identity)
                return None

            try:
                read_result = self._read_payload_bytes_secure()
            except OSError as exc:
                if exc.errno in {errno.ELOOP, getattr(errno, "EMLINK", errno.ELOOP)}:
                    _LOGGER.warning(
                        "Refusing to follow symlinked display ambient impulse cue at %s.",
                        self.path,
                    )
                    self._set_cache_invalid(identity)
                    return None
                _LOGGER.warning("Failed to read display ambient impulse cue from %s.", self.path, exc_info=True)
                self._set_cache_invalid(identity)
                return None

            if read_result is None:
                self._set_cache_missing()
                return None

            raw, file_stat = read_result
            file_identity = _stat_identity(file_stat)

            if not stat.S_ISREG(file_stat.st_mode):
                _LOGGER.warning(
                    "Ignoring display ambient impulse cue at %s because the opened artifact is not a regular file.",
                    self.path,
                )
                self._set_cache_invalid(file_identity)
                return None

            if len(raw) > self.max_bytes:
                _LOGGER.warning(
                    "Ignoring display ambient impulse cue at %s because the opened payload exceeds %s bytes.",
                    self.path,
                    self.max_bytes,
                )
                self._set_cache_invalid(file_identity)
                return None

            try:
                payload = json.loads(raw)
            except Exception:
                _LOGGER.warning("Failed to parse display ambient impulse cue from %s.", self.path, exc_info=True)
                self._set_cache_invalid(file_identity)
                return None

            if not isinstance(payload, dict):
                _LOGGER.warning(
                    "Ignoring invalid display ambient impulse cue payload at %s because it is not an object.",
                    self.path,
                )
                self._set_cache_invalid(file_identity)
                return None

            fallback_updated_at = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)
            try:
                cue = DisplayAmbientImpulseCue.from_dict(
                    payload,
                    fallback_updated_at=fallback_updated_at,
                    default_ttl_s=self.default_ttl_s,
                    max_ttl_s=self.max_ttl_s,
                )
            except Exception:
                _LOGGER.warning("Ignoring invalid display ambient impulse cue payload at %s.", self.path, exc_info=True)
                self._set_cache_invalid(file_identity)
                return None

            self._set_cache_valid(file_identity, cue)
            return cue

    def load_active(self, *, now: datetime | None = None) -> DisplayAmbientImpulseCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayAmbientImpulseCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayAmbientImpulseCue:
        """Persist one ambient impulse cue with a bounded expiry."""

        with self._thread_lock:
            written_at = (now or _utc_now()).astimezone(timezone.utc)
            ttl_s = _coerce_ttl_s(
                self.default_ttl_s if hold_seconds is None else hold_seconds,
                default=self.default_ttl_s,
                max_ttl_s=self.max_ttl_s,
            )
            normalized = DisplayAmbientImpulseCue(
                source=cue.source,
                updated_at=_format_timestamp(written_at),
                expires_at=_format_timestamp(written_at + timedelta(seconds=ttl_s)),
                topic_key=cue.topic_key,
                semantic_topic_key=cue.semantic_topic_key,
                eyebrow=cue.eyebrow,
                headline=cue.headline,
                body=cue.body,
                symbol=cue.symbol,
                accent=cue.accent,
                action=cue.action,
                attention_state=cue.attention_state,
            )
            payload_bytes = (
                json.dumps(
                    self._build_store_payload(normalized),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                ).encode("utf-8")
                + b"\n"
            )

            with self._writer_lock():
                self._atomic_write_payload_bytes(payload_bytes)
                try:
                    file_stat = os.stat(self.path, follow_symlinks=False)
                except OSError:
                    self._set_cache_missing()
                else:
                    self._set_cache_valid(_stat_identity(file_stat), normalized)
            return normalized

    def clear(self) -> None:
        """Remove the persisted cue artifact when it exists."""

        with self._thread_lock:
            try:
                with self._writer_lock():
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    else:
                        _fsync_directory(self.path.parent)
            except OSError:
                _LOGGER.warning("Failed to clear display ambient impulse cue at %s.", self.path, exc_info=True)
            finally:
                self._set_cache_missing()


@dataclass(slots=True)
class DisplayAmbientImpulseController:
    """Persist short-lived ambient impulses for the HDMI reserve card."""

    store: DisplayAmbientImpulseCueStore
    default_source: str = "ambient_impulse"

    @classmethod
    def from_config(
        cls,
        config: _DisplayConfigLike,
        *,
        default_source: str = "ambient_impulse",
    ) -> "DisplayAmbientImpulseController":
        """Build one controller from the configured cue store."""

        return cls(
            store=DisplayAmbientImpulseCueStore.from_config(config),
            default_source=_compact_text(default_source, max_len=80) or "ambient_impulse",
        )

    def show_impulse(
        self,
        *,
        topic_key: str,
        semantic_topic_key: str | None = None,
        eyebrow: str,
        headline: str,
        body: str,
        symbol: DisplayEmojiSymbol | str = DisplayEmojiSymbol.SPARKLES,
        accent: str = "info",
        action: str = "hint",
        attention_state: str = "background",
        hold_seconds: float | None = None,
        source: str | None = None,
        now: datetime | None = None,
    ) -> DisplayAmbientImpulseCue:
        """Persist one active ambient impulse cue."""

        cue = DisplayAmbientImpulseCue(
            source=_compact_text(source, max_len=80) or self.default_source,
            topic_key=topic_key,
            semantic_topic_key=_compact_text(semantic_topic_key, max_len=96).casefold(),
            eyebrow=eyebrow,
            headline=headline,
            body=body,
            symbol=_normalize_symbol(symbol).value,
            accent=accent,
            action=action,
            attention_state=attention_state,
        )
        return self.store.save(
            cue,
            hold_seconds=hold_seconds,
            now=now,
        )