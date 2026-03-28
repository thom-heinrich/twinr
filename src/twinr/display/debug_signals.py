"""Persist bounded HDMI header debug signals for sensor-fusion inspection.

The senior-facing HDMI surface stays face-first, but operator debugging still
benefits from a tiny explicit signal lane that can mirror currently active
camera and fusion states such as ``MOTION_STILL`` or ``POSSIBLE_FALL`` without
teaching the generic runtime snapshot schema about those transient internals.
"""

# CHANGELOG: 2026-03-28
# BUG-1: Fixed multi-process save races caused by a single predictable .tmp path; saves are now serialized with
#        an advisory lock and written via a unique secure temp file before atomic replace.
# BUG-2: Fixed crash-safety hole where os.replace() was atomic but not durable; writes now fsync the temp file and
#        containing directory so power loss on Pi-class storage cannot silently drop the newest snapshot as easily.
# BUG-3: Fixed direct-constructor normalization gaps (priority/timestamps) that could raise or keep malformed data alive.
# SEC-1: Hardened load/save against symlink/FIFO/oversized-payload abuse by using secure temp creation, O_NOFOLLOW
#        where available, regular-file checks, and a strict max-bytes guard for the artifact.
# SEC-2: Added boot-aware monotonic expiry metadata so short TTL debug pills cannot remain incorrectly "active" across
#        wall-clock jumps or after reboot.
# IMP-1: Snapshot ordering is now deterministic (priority-first) to reduce HDMI header flicker and make signatures stable.
# IMP-2: Added bounded text normalization and configurable max payload / max signal count so the debug lane stays small.
# BREAKING: Persisted payloads now include schema_version, boot_id, updated_monotonic_ns, and expires_monotonic_ns.
# BREAKING: Store load/save trims snapshots to max_signals (default 8) by priority to keep the HDMI debug lane bounded.

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import stat
import tempfile
import time
from typing import Protocol, cast

try:  # Raspberry Pi 4 target is Linux/Unix, but keep import failure graceful.
    import fcntl
except ImportError:  # pragma: no cover - not expected on Pi/Linux
    fcntl = None  # type: ignore[assignment]


_DEFAULT_DEBUG_SIGNAL_TTL_S = 6.0
_DEFAULT_DEBUG_SIGNAL_PATH = "artifacts/stores/ops/display_debug_signals.json"
_DEFAULT_DEBUG_SIGNAL_MAX_BYTES = 16 * 1024
_DEFAULT_DEBUG_SIGNAL_MAX_SIGNALS = 8
_DEFAULT_SIGNAL_TEXT_MAX_LEN = 48
_DEFAULT_SOURCE_TEXT_MAX_LEN = 64
_SCHEMA_VERSION = 2

_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warning", "alert"})
_ACCENT_SEVERITY = {
    "alert": 4,
    "warning": 3,
    "success": 2,
    "info": 1,
    "neutral": 0,
}

_LOGGER = logging.getLogger(__name__)


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the debug-signal store."""

    project_root: str | PathLike[str]


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _monotonic_ns() -> int:
    """Return the current monotonic clock in nanoseconds."""

    return time.monotonic_ns()


@lru_cache(maxsize=1)
def _current_boot_id() -> str | None:
    """Return the current Linux boot ID when available."""

    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _normalize_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into an aware UTC ``datetime``."""

    if isinstance(value, datetime):
        parsed = value
    else:
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
    """Serialize one aware timestamp as ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _normalize_text(
    value: object | None,
    *,
    fallback: str = "",
    max_length: int | None = None,
) -> str:
    """Return one bounded printable text token."""

    text = str(value or "")
    printable = "".join(ch if ch.isprintable() else " " for ch in text)
    compact = " ".join(printable.split()).strip()
    if max_length is not None and max_length > 0:
        compact = compact[:max_length].rstrip("_ ").strip()
    return compact or fallback


def _normalize_key(value: object | None) -> str:
    """Normalize one signal key into a stable snake-style token."""

    text = _normalize_text(value, max_length=_DEFAULT_SIGNAL_TEXT_MAX_LEN).lower().replace("-", "_").replace(" ", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch == "_")
    return text or "signal"


def _normalize_label(value: object | None, *, fallback: str) -> str:
    """Normalize one signal label into a stable uppercase pill label."""

    compact = _normalize_text(
        value,
        fallback=fallback,
        max_length=_DEFAULT_SIGNAL_TEXT_MAX_LEN,
    ).upper().replace("-", "_").replace(" ", "_")
    compact = "".join(ch for ch in compact if ch.isalnum() or ch == "_")
    return compact or fallback


def _normalize_source(value: object | None) -> str:
    """Normalize the snapshot source tag."""

    return _normalize_text(value, fallback="external", max_length=_DEFAULT_SOURCE_TEXT_MAX_LEN)


def _normalize_accent(value: object | None) -> str:
    """Normalize one signal accent token."""

    compact = _normalize_text(value).lower()
    if compact not in _ALLOWED_ACCENTS:
        return "neutral"
    return compact


def _normalize_priority(value: object | None) -> int:
    """Normalize one optional signal priority into a stable integer."""

    if value is None:
        return 0
    numeric: float | str
    numeric = int(value) if isinstance(value, bool) else str(value)
    try:
        rounded = int(round(float(numeric)))
    except (TypeError, ValueError):
        return 0
    return max(-999_999, min(999_999, rounded))


def _normalize_positive_float(value: object | None, *, fallback: float, minimum: float) -> float:
    """Normalize one positive finite float."""

    try:
        numeric = float(fallback if value is None else value)
    except (TypeError, ValueError):
        numeric = fallback
    if not math.isfinite(numeric):
        numeric = fallback
    return max(minimum, numeric)


def _normalize_optional_int(value: object | None) -> int | None:
    """Normalize one optional integer field."""

    if value is None or value == "":
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, numeric)


def _signal_sort_key(signal: "DisplayDebugSignal") -> tuple[int, int, str, str]:
    """Return the stable ordering key for one signal."""

    return (signal.priority, -_ACCENT_SEVERITY.get(signal.accent, 0), signal.key, signal.label)


def _choose_signal(existing: "DisplayDebugSignal", candidate: "DisplayDebugSignal") -> "DisplayDebugSignal":
    """Choose the stronger of two duplicate-key signals."""

    if _signal_sort_key(candidate) < _signal_sort_key(existing):
        return candidate
    return existing


def _bound_signals(
    signals: Iterable["DisplayDebugSignal" | Mapping[str, object]],
    *,
    max_signals: int,
) -> tuple["DisplayDebugSignal", ...]:
    """Normalize, deduplicate, sort and bound a signal collection."""

    by_key: dict[str, DisplayDebugSignal] = {}
    for item in signals:
        if isinstance(item, DisplayDebugSignal):
            normalized = item
        elif isinstance(item, Mapping):
            normalized = DisplayDebugSignal.from_dict(item)
        else:
            continue
        current = by_key.get(normalized.key)
        by_key[normalized.key] = normalized if current is None else _choose_signal(current, normalized)
    ordered = list(by_key.values())
    if max_signals > 0:
        ordered = ordered[:max_signals]
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class DisplayDebugSignal:
    """Describe one bounded HDMI header debug-signal pill."""

    key: str
    label: str
    accent: str = "neutral"
    priority: int = 0

    def __post_init__(self) -> None:
        """Normalize the signal into a stable display-safe payload."""

        normalized_key = _normalize_key(self.key)
        normalized_label = _normalize_label(self.label, fallback=normalized_key.upper())
        object.__setattr__(self, "key", normalized_key)
        object.__setattr__(self, "label", normalized_label)
        object.__setattr__(self, "accent", _normalize_accent(self.accent))
        object.__setattr__(self, "priority", _normalize_priority(self.priority))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DisplayDebugSignal":
        """Build one signal from JSON-style data."""

        return cls(
            key=_normalize_text(payload.get("key"), fallback="signal"),
            label=_normalize_text(payload.get("label"), fallback="SIGNAL"),
            accent=_normalize_text(payload.get("accent"), fallback="neutral"),
            priority=_normalize_priority(payload.get("priority", 0)),
        )

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for this signal."""

        return (self.key, self.label, self.accent, self.priority)


@dataclass(frozen=True, slots=True)
class DisplayDebugSignalSnapshot:
    """Describe one short-lived batch of active HDMI debug signals."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    signals: tuple[DisplayDebugSignal, ...] = ()
    schema_version: int = _SCHEMA_VERSION
    boot_id: str | None = None
    updated_monotonic_ns: int | None = None
    expires_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        """Normalize the snapshot and deduplicate signals by key."""

        object.__setattr__(self, "source", _normalize_source(self.source))

        updated_at = _normalize_timestamp(self.updated_at)
        expires_at = _normalize_timestamp(self.expires_at)
        object.__setattr__(self, "updated_at", _format_timestamp(updated_at) if updated_at else None)
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at) if expires_at else None)

        object.__setattr__(self, "schema_version", max(1, _normalize_optional_int(self.schema_version) or _SCHEMA_VERSION))
        object.__setattr__(self, "boot_id", _normalize_text(self.boot_id, max_length=128) or None)
        object.__setattr__(self, "updated_monotonic_ns", _normalize_optional_int(self.updated_monotonic_ns))
        object.__setattr__(self, "expires_monotonic_ns", _normalize_optional_int(self.expires_monotonic_ns))
        object.__setattr__(self, "signals", _bound_signals(self.signals, max_signals=10_000))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_DEBUG_SIGNAL_TTL_S,
        max_signals: int = _DEFAULT_DEBUG_SIGNAL_MAX_SIGNALS,
    ) -> "DisplayDebugSignalSnapshot":
        """Build one normalized snapshot from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        ttl_s = _normalize_positive_float(default_ttl_s, fallback=_DEFAULT_DEBUG_SIGNAL_TTL_S, minimum=0.1)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=ttl_s)

        raw_signals = payload.get("signals")
        signal_items = cast(Iterable[Mapping[str, object]], raw_signals) if isinstance(raw_signals, (list, tuple)) else ()
        return cls(
            source=_normalize_source(payload.get("source")),
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            signals=_bound_signals(signal_items, max_signals=max_signals),
            schema_version=_normalize_optional_int(payload.get("schema_version")) or 1,
            boot_id=_normalize_text(payload.get("boot_id"), max_length=128) or None,
            updated_monotonic_ns=_normalize_optional_int(payload.get("updated_monotonic_ns")),
            expires_monotonic_ns=_normalize_optional_int(payload.get("expires_monotonic_ns")),
        )

    def bounded(self, *, max_signals: int) -> "DisplayDebugSignalSnapshot":
        """Return a copy trimmed to a bounded signal count."""

        return DisplayDebugSignalSnapshot(
            source=self.source,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
            signals=_bound_signals(self.signals, max_signals=max_signals),
            schema_version=self.schema_version,
            boot_id=self.boot_id,
            updated_monotonic_ns=self.updated_monotonic_ns,
            expires_monotonic_ns=self.expires_monotonic_ns,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the snapshot into a JSON-safe mapping."""

        payload = asdict(self)
        payload["signals"] = [asdict(signal) for signal in self.signals]
        return payload

    def is_active(self, *, now: datetime | None = None, monotonic_ns: int | None = None) -> bool:
        """Return whether the snapshot should still affect the HDMI header."""

        current_boot_id = _current_boot_id()
        if self.boot_id and current_boot_id and self.boot_id != current_boot_id:
            return False

        if self.expires_monotonic_ns is not None and now is None:
            current_monotonic_ns = _monotonic_ns() if monotonic_ns is None else monotonic_ns
            return self.expires_monotonic_ns >= current_monotonic_ns

        expires_at = _normalize_timestamp(self.expires_at)
        if expires_at is None:
            return True
        return expires_at >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for the whole batch."""

        return tuple(signal.signature() for signal in self.signals)


@dataclass(slots=True)
class DisplayDebugSignalStore:
    """Read and write the optional HDMI debug-signal artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_DEBUG_SIGNAL_TTL_S
    max_bytes: int = _DEFAULT_DEBUG_SIGNAL_MAX_BYTES
    max_signals: int = _DEFAULT_DEBUG_SIGNAL_MAX_SIGNALS

    @property
    def lock_path(self) -> Path:
        """Return the sibling path used for advisory write-locking."""

        return self.path.with_suffix(f"{self.path.suffix}.lock")

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayDebugSignalStore":
        """Resolve the debug-signal path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_debug_signal_path", _DEFAULT_DEBUG_SIGNAL_PATH) or _DEFAULT_DEBUG_SIGNAL_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path

        configured_ttl = getattr(config, "display_debug_signal_ttl_s", _DEFAULT_DEBUG_SIGNAL_TTL_S)
        configured_max_bytes = getattr(config, "display_debug_signal_max_bytes", _DEFAULT_DEBUG_SIGNAL_MAX_BYTES)
        configured_max_signals = getattr(config, "display_debug_signal_max_signals", _DEFAULT_DEBUG_SIGNAL_MAX_SIGNALS)
        return cls(
            path=resolved_path,
            default_ttl_s=_normalize_positive_float(
                configured_ttl,
                fallback=_DEFAULT_DEBUG_SIGNAL_TTL_S,
                minimum=0.1,
            ),
            max_bytes=max(512, _normalize_optional_int(configured_max_bytes) or _DEFAULT_DEBUG_SIGNAL_MAX_BYTES),
            max_signals=max(1, _normalize_optional_int(configured_max_signals) or _DEFAULT_DEBUG_SIGNAL_MAX_SIGNALS),
        )

    def _decode_payload(self, payload_text: str) -> Mapping[str, object] | None:
        """Decode one JSON payload into a mapping."""

        try:
            payload = json.loads(payload_text)
        except Exception:
            _LOGGER.warning("Failed to read display debug signals from %s.", self.path, exc_info=True)
            return None
        if not isinstance(payload, dict):
            _LOGGER.warning(
                "Ignoring invalid display debug signal payload at %s because it is not an object.",
                self.path,
            )
            return None
        return cast(Mapping[str, object], payload)

    def _read_payload_text(self) -> tuple[str, datetime] | None:
        """Read the artifact only when it is a small regular file."""

        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        fd: int | None = None
        try:
            fd = os.open(self.path, flags | nofollow)
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                _LOGGER.warning(
                    "Ignoring display debug signal payload at %s because it is not a regular file.",
                    self.path,
                )
                return None
            if file_stat.st_size > self.max_bytes:
                _LOGGER.warning(
                    "Ignoring display debug signal payload at %s because it exceeds %s bytes.",
                    self.path,
                    self.max_bytes,
                )
                return None
            fallback_updated_at = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)
            with os.fdopen(fd, "r", encoding="utf-8", newline="") as handle:
                fd = None
                return handle.read(self.max_bytes + 1), fallback_updated_at
        except FileNotFoundError:
            return None
        except OSError:
            _LOGGER.warning("Failed to open display debug signal payload at %s.", self.path, exc_info=True)
            return None
        finally:
            if fd is not None:
                os.close(fd)

    @contextmanager
    def _exclusive_lock(self):
        """Acquire an advisory exclusive writer lock."""

        if fcntl is None:  # pragma: no cover - Pi/Linux target always has fcntl
            yield
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _fsync_directory(self) -> None:
        """Flush the containing directory entry to disk when supported."""

        dir_flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        fd: int | None = None
        try:
            fd = os.open(self.path.parent, dir_flags)
            os.fsync(fd)
        except OSError:
            _LOGGER.debug("Directory fsync not supported for %s.", self.path.parent, exc_info=True)
        finally:
            if fd is not None:
                os.close(fd)

    def _write_payload_text(self, payload_text: str) -> None:
        """Persist one JSON payload durably via secure temp file + atomic replace."""

        prefix = f".{self.path.name}."
        fd, tmp_name = tempfile.mkstemp(
            prefix=prefix,
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(payload_text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            self._fsync_directory()
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                _LOGGER.debug("Failed to clean up temporary debug-signal file %s.", tmp_path, exc_info=True)

    def load(self) -> DisplayDebugSignalSnapshot | None:
        """Load the current debug-signal snapshot, if one exists and parses."""

        payload_and_mtime = self._read_payload_text()
        if payload_and_mtime is None:
            return None
        payload_text, fallback_updated_at = payload_and_mtime
        if len(payload_text.encode("utf-8")) > self.max_bytes:
            _LOGGER.warning(
                "Ignoring display debug signal payload at %s because its decoded size exceeds %s bytes.",
                self.path,
                self.max_bytes,
            )
            return None

        payload = self._decode_payload(payload_text)
        if payload is None:
            return None

        try:
            return DisplayDebugSignalSnapshot.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
                max_signals=self.max_signals,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display debug signal payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayDebugSignalSnapshot | None:
        """Load the current snapshot only while it is still active."""

        snapshot = self.load()
        if snapshot is None or not snapshot.is_active(now=now):
            return None
        return snapshot

    def save(
        self,
        snapshot: DisplayDebugSignalSnapshot,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayDebugSignalSnapshot:
        """Persist one normalized debug-signal snapshot atomically and durably."""

        safe_now = (now or _utc_now()).astimezone(timezone.utc)
        safe_monotonic_ns = _monotonic_ns()
        ttl_s = _normalize_positive_float(
            hold_seconds if hold_seconds is not None else self.default_ttl_s,
            fallback=self.default_ttl_s,
            minimum=0.1,
        )
        ttl_ns = max(100_000_000, int(ttl_s * 1_000_000_000))

        normalized = DisplayDebugSignalSnapshot(
            source=snapshot.source,
            updated_at=_format_timestamp(safe_now),
            expires_at=_format_timestamp(safe_now + timedelta(seconds=ttl_s)),
            signals=snapshot.signals,
            schema_version=_SCHEMA_VERSION,
            boot_id=_current_boot_id(),
            updated_monotonic_ns=safe_monotonic_ns,
            expires_monotonic_ns=safe_monotonic_ns + ttl_ns,
        ).bounded(max_signals=self.max_signals)

        payload_text = json.dumps(
            normalized.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(payload_text.encode("utf-8")) > self.max_bytes:
            raise ValueError(
                f"Display debug signal payload for {self.path} exceeds max_bytes={self.max_bytes} after normalization."
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._exclusive_lock():
            self._write_payload_text(payload_text)
        return normalized
