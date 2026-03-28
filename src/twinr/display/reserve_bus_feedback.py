from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import errno
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback; Raspberry Pi uses Linux.
    fcntl = None  # type: ignore[assignment]

# CHANGELOG: 2026-03-28
# BUG-1: Invalid or partially written JSON could be normalized into a fresh active signal because
#        missing/invalid timestamps defaulted to "now" and unknown reactions defaulted to
#        "engaged" during load.
# BUG-2: Concurrent writers could overwrite a newer signal with an older one, and direct writes
#        could leave truncated/corrupt JSON on crash or power loss.
# BUG-3: record_reaction() could crash on invalid/NaN/inf TTL input and treated naive datetimes
#        ambiguously.
# SEC-1: Plain write_text() followed symlinks and respected ambient umask, which made local file
#        clobbering/tampering materially easier on a Pi deployment if the artifact path became
#        writable by another local user or process.
# SEC-2: Unbounded reads allowed local denial-of-service by replacing the artifact with a very
#        large file.
# IMP-1: Added atomic same-directory writes with fsync, advisory locking, restrictive file modes,
#        stale-write suppression, and bounded no-symlink reads.
# IMP-2: Added strict on-disk schema validation, config-driven TTL overrides, and frontier-style
#        smooth decay helpers (remaining_seconds / decayed_intensity / bias_score) for adaptive
#        short-term planning.

_SCHEMA_VERSION = 2
_DEFAULT_FEEDBACK_PATH = "artifacts/stores/ops/display_reserve_bus_feedback.json"
_DEFAULT_FEEDBACK_TTL_S = 4.0 * 60.0 * 60.0
_MIN_FEEDBACK_TTL_S = 60.0
_MAX_FEEDBACK_TTL_S = 24.0 * 60.0 * 60.0
# BREAKING: explicit TTLs above 24h are now clamped because this store is defined as short-lived
#           reserve-bus state for today's plan, not multi-day memory.
_MAX_FEEDBACK_BYTES = 16 * 1024
_FILE_MODE = 0o600
_LOCK_SUFFIX = ".lock"
_ALLOWED_REACTIONS = frozenset(
    {
        "immediate_engagement",
        "engaged",
        "cooled",
        "avoided",
        "ignored",
    }
)
_REACTION_WEIGHTS = {
    "immediate_engagement": 1.25,
    "engaged": 1.0,
    "cooled": -0.25,
    "avoided": -0.75,
    "ignored": -0.5,
}

_LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Serialize one aware timestamp as UTC ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _normalize_datetime(value: datetime | None, *, default_now: bool = False) -> datetime | None:
    """Normalize one optional datetime into an aware UTC timestamp."""

    if value is None:
        return _utc_now() if default_now else None
    if value.tzinfo is None:
        _LOGGER.warning(
            "Naive datetime passed to display reserve bus feedback store; interpreting it as UTC."
        )
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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


def _compact_text(value: object | None, *, max_len: int) -> str:
    """Collapse one arbitrary value into bounded single-line text."""

    if value is None:
        return ""
    compact = " ".join(str(value).split()).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_reaction(value: object | None, *, strict: bool = False) -> str:
    """Normalize one bounded reserve-bus reaction token."""

    compact = _compact_text(value, max_len=40).lower().replace("-", "_").replace(" ", "_")
    if compact in _ALLOWED_REACTIONS:
        return compact
    if strict:
        raise ValueError(f"Unsupported reserve-bus reaction: {compact or '<empty>'}.")
    return "engaged"


def _normalize_intensity(value: object | None) -> float:
    """Clamp one optional feedback intensity into the inclusive 0..1 range."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _normalize_ttl_seconds(value: object | None, *, default: float) -> float:
    """Clamp one TTL-like value into a safe short-lived duration."""

    try:
        ttl_s = float(value if value is not None else default)
    except (TypeError, ValueError):
        ttl_s = default
    if not math.isfinite(ttl_s):
        ttl_s = default
    return max(_MIN_FEEDBACK_TTL_S, min(_MAX_FEEDBACK_TTL_S, ttl_s))


def _normalize_max_bytes(value: object | None, *, default: int = _MAX_FEEDBACK_BYTES) -> int:
    """Clamp the maximum readable artifact size into a small safe bound."""

    try:
        number = int(value if value is not None else default)
    except (TypeError, ValueError):
        number = default
    return max(1024, min(256 * 1024, number))


@contextmanager
def _locked_file(path: Path) -> Iterator[Any]:
    """Open one lock file and hold an exclusive advisory lock for the context."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, _FILE_MODE)
    handle = os.fdopen(fd, "a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            os.fchmod(handle.fileno(), _FILE_MODE)
        except (AttributeError, OSError):
            pass
        yield handle
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of one directory after rename/unlink style operations."""

    if os.name != "posix":  # pragma: no cover - Raspberry Pi uses POSIX.
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@dataclass(frozen=True, slots=True)
class DisplayReserveBusFeedbackSignal:
    """Describe one short-lived reserve-bus feedback hint.

    Attributes:
        schema_version: On-disk schema version for strict validation and upgrades.
        source: Producer of the feedback hint.
        requested_at: UTC timestamp when the hint was recorded.
        expires_at: UTC timestamp after which the hint should no longer bias
            reserve planning.
        topic_key: Stable normalized topic key this hint refers to.
        reaction: Coarse reaction token such as ``immediate_engagement``.
        intensity: Relative strength of the reaction in the 0..1 range.
        reason: Short auditable explanation for the hint.
    """

    schema_version: int = _SCHEMA_VERSION
    source: str = "display_reserve_card"
    requested_at: str | None = None
    expires_at: str | None = None
    topic_key: str = ""
    reaction: str = "engaged"
    intensity: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical contract."""

        object.__setattr__(self, "schema_version", int(self.schema_version or _SCHEMA_VERSION))
        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "display_reserve_card")
        object.__setattr__(self, "topic_key", _compact_text(self.topic_key, max_len=96).casefold())
        # BREAKING: invalid reactions now raise at construction instead of silently collapsing to
        #           "engaged", because that produced false positive bias under upstream typos.
        object.__setattr__(self, "reaction", _normalize_reaction(self.reaction, strict=True))
        object.__setattr__(self, "intensity", _normalize_intensity(self.intensity))
        requested_at_text = _compact_text(self.requested_at, max_len=64)
        expires_at_text = _compact_text(self.expires_at, max_len=64)
        requested_at = _normalize_timestamp(requested_at_text) if requested_at_text else _utc_now()
        if requested_at_text and requested_at is None:
            # BREAKING: malformed requested_at values now raise instead of silently becoming "now".
            raise ValueError("Invalid requested_at timestamp for display reserve bus feedback")
        expires_at = (
            _normalize_timestamp(expires_at_text)
            if expires_at_text
            else requested_at + timedelta(seconds=_DEFAULT_FEEDBACK_TTL_S)
        )
        if expires_at_text and expires_at is None:
            # BREAKING: malformed expires_at values now raise instead of silently extending TTL.
            raise ValueError("Invalid expires_at timestamp for display reserve bus feedback")
        if expires_at < requested_at:
            expires_at = requested_at
        object.__setattr__(self, "requested_at", _format_timestamp(requested_at))
        object.__setattr__(self, "expires_at", _format_timestamp(expires_at))
        object.__setattr__(self, "reason", _compact_text(self.reason, max_len=160))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        strict: bool = False,
    ) -> "DisplayReserveBusFeedbackSignal":
        """Build one feedback hint from persisted JSON-style data."""

        schema_version_raw = payload.get("schema_version", _SCHEMA_VERSION)
        try:
            schema_version = int(schema_version_raw)
        except (TypeError, ValueError):
            if strict:
                raise ValueError("Invalid schema_version in reserve-bus feedback payload.") from None
            schema_version = _SCHEMA_VERSION
        source = _compact_text(payload.get("source"), max_len=80) or "display_reserve_card"
        topic_key = _compact_text(payload.get("topic_key"), max_len=96).casefold()
        reaction = _normalize_reaction(payload.get("reaction"), strict=strict)
        intensity = _normalize_intensity(payload.get("intensity"))
        reason = _compact_text(payload.get("reason"), max_len=160)
        requested_at = _normalize_timestamp(payload.get("requested_at"))
        expires_at = _normalize_timestamp(payload.get("expires_at"))

        if strict:
            if schema_version <= 0:
                raise ValueError("Invalid schema_version in reserve-bus feedback payload.")
            if not topic_key:
                raise ValueError("Reserve-bus feedback payload requires a non-empty topic_key.")
            if requested_at is None:
                raise ValueError("Reserve-bus feedback payload has an invalid requested_at timestamp.")
            if expires_at is None:
                raise ValueError("Reserve-bus feedback payload has an invalid expires_at timestamp.")
            if expires_at < requested_at:
                raise ValueError("Reserve-bus feedback payload expires before it was requested.")

        return cls(
            schema_version=schema_version,
            source=source,
            requested_at=_format_timestamp(requested_at) if requested_at is not None else None,
            expires_at=_format_timestamp(expires_at) if expires_at is not None else None,
            topic_key=topic_key,
            reaction=reaction,
            intensity=intensity,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the feedback hint into JSON-safe data."""

        return asdict(self)

    def requested_at_datetime(self) -> datetime:
        """Return the requested-at timestamp as an aware UTC datetime."""

        return _normalize_timestamp(self.requested_at) or _utc_now()

    def expires_at_datetime(self) -> datetime:
        """Return the expiry timestamp as an aware UTC datetime."""

        return _normalize_timestamp(self.expires_at) or self.requested_at_datetime()

    def ttl_seconds(self) -> float:
        """Return the signal lifetime in seconds."""

        return max(0.0, (self.expires_at_datetime() - self.requested_at_datetime()).total_seconds())

    def age_seconds(self, *, now: datetime | None = None) -> float:
        """Return how old the signal is in seconds."""

        effective_now = _normalize_datetime(now, default_now=True) or _utc_now()
        return max(0.0, (effective_now - self.requested_at_datetime()).total_seconds())

    def remaining_seconds(self, *, now: datetime | None = None) -> float:
        """Return the remaining lifetime in seconds."""

        effective_now = _normalize_datetime(now, default_now=True) or _utc_now()
        return max(0.0, (self.expires_at_datetime() - effective_now).total_seconds())

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether this hint should still bias reserve planning."""

        return self.remaining_seconds(now=now) > 0.0

    def decayed_intensity(self, *, now: datetime | None = None) -> float:
        """Return a smoothly decayed intensity for adaptive short-term planning."""

        ttl_s = self.ttl_seconds()
        if ttl_s <= 0.0:
            return 0.0
        remaining_fraction = self.remaining_seconds(now=now) / ttl_s
        return max(0.0, min(1.0, self.intensity * remaining_fraction))

    def bias_score(self, *, now: datetime | None = None) -> float:
        """Return one signed planning bias score with smooth temporal decay."""

        return self.decayed_intensity(now=now) * _REACTION_WEIGHTS.get(self.reaction, 0.0)


@dataclass(slots=True)
class DisplayReserveBusFeedbackStore:
    """Read and write one short-lived reserve-bus feedback hint."""

    path: Path
    default_ttl_s: float = _DEFAULT_FEEDBACK_TTL_S
    max_bytes: int = _MAX_FEEDBACK_BYTES

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayReserveBusFeedbackStore":
        """Resolve the feedback artifact path from configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured = Path(
            getattr(config, "display_reserve_bus_feedback_path", _DEFAULT_FEEDBACK_PATH)
            or _DEFAULT_FEEDBACK_PATH
        )
        resolved = configured if configured.is_absolute() else project_root / configured
        default_ttl_s = _normalize_ttl_seconds(
            getattr(config, "display_reserve_bus_feedback_ttl_s", _DEFAULT_FEEDBACK_TTL_S),
            default=_DEFAULT_FEEDBACK_TTL_S,
        )
        max_bytes = _normalize_max_bytes(
            getattr(config, "display_reserve_bus_feedback_max_bytes", _MAX_FEEDBACK_BYTES),
            default=_MAX_FEEDBACK_BYTES,
        )
        return cls(path=resolved, default_ttl_s=default_ttl_s, max_bytes=max_bytes)

    @property
    def lock_path(self) -> Path:
        """Return the sidecar lock path used for cross-process coordination."""

        return self.path.with_name(f".{self.path.name}{_LOCK_SUFFIX}")

    def _validate_persistable_signal(self, signal: DisplayReserveBusFeedbackSignal) -> None:
        """Reject signals that would be ambiguous or dangerous to persist."""

        # BREAKING: blank topic keys are now rejected instead of being silently persisted because
        #           they make downstream planning semantics ambiguous.
        if not signal.topic_key:
            raise ValueError("display reserve bus feedback requires a non-empty topic_key")
        if signal.expires_at_datetime() < signal.requested_at_datetime():
            raise ValueError("display reserve bus feedback expires before it was requested")

    def _read_payload_unlocked(self) -> Mapping[str, object] | None:
        """Read one JSON payload with bounded size and no symlink traversal."""

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self.path, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOENT}:
                _LOGGER.warning(
                    "Refusing to read display reserve bus feedback from unsafe path %s.",
                    self.path,
                )
                return None
            _LOGGER.warning(
                "Failed to open display reserve bus feedback from %s.",
                self.path,
                exc_info=True,
            )
            return None
        try:
            stat_result = os.fstat(fd)
            if stat_result.st_size > self.max_bytes:
                _LOGGER.warning(
                    "Ignoring display reserve bus feedback at %s because it exceeds %d bytes.",
                    self.path,
                    self.max_bytes,
                )
                return None
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = -1
                payload = json.load(handle)
        except Exception:
            _LOGGER.warning("Failed to read display reserve bus feedback from %s.", self.path, exc_info=True)
            return None
        finally:
            if fd >= 0:
                os.close(fd)
        if not isinstance(payload, Mapping):
            _LOGGER.warning(
                "Ignoring invalid display reserve bus feedback payload at %s because it is not an object.",
                self.path,
            )
            return None
        return payload

    def _load_unlocked(self, *, strict: bool) -> DisplayReserveBusFeedbackSignal | None:
        """Load one signal without taking the store lock."""

        payload = self._read_payload_unlocked()
        if payload is None:
            return None
        try:
            signal = DisplayReserveBusFeedbackSignal.from_dict(payload, strict=strict)
        except Exception:
            _LOGGER.warning("Ignoring invalid display reserve bus feedback payload at %s.", self.path, exc_info=True)
            return None
        try:
            self._validate_persistable_signal(signal)
        except Exception:
            _LOGGER.warning("Ignoring unsafe display reserve bus feedback payload at %s.", self.path, exc_info=True)
            return None
        return signal

    def load(self) -> DisplayReserveBusFeedbackSignal | None:
        """Load the current reserve-bus feedback hint, if present."""

        return self._load_unlocked(strict=True)

    def load_active(self, *, now: datetime | None = None) -> DisplayReserveBusFeedbackSignal | None:
        """Load the current hint only when it is still active."""

        signal = self.load()
        if signal is None:
            return None
        if not signal.is_active(now=now):
            return None
        return signal

    def _write_signal_unlocked(self, signal: DisplayReserveBusFeedbackSignal) -> DisplayReserveBusFeedbackSignal:
        """Persist one signal atomically while holding the store lock."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(signal.to_dict(), handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(tmp_path, _FILE_MODE)
            except OSError:
                pass
            os.replace(tmp_path, self.path)
            try:
                os.chmod(self.path, _FILE_MODE)
            except OSError:
                pass
            _fsync_directory(self.path.parent)
        finally:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
        return signal

    def save(self, signal: DisplayReserveBusFeedbackSignal) -> DisplayReserveBusFeedbackSignal:
        """Persist one reserve-bus feedback hint."""

        self._validate_persistable_signal(signal)
        with _locked_file(self.lock_path):
            existing = self._load_unlocked(strict=True)
            if existing is not None and existing.requested_at_datetime() > signal.requested_at_datetime():
                _LOGGER.info(
                    "Ignoring stale display reserve bus feedback for %s because a newer signal already exists.",
                    signal.topic_key,
                )
                return existing
            return self._write_signal_unlocked(signal)

    def clear(self) -> None:
        """Remove the persisted feedback artifact when it exists."""

        with _locked_file(self.lock_path):
            try:
                self.path.unlink()
            except FileNotFoundError:
                return
            except OSError:
                _LOGGER.warning("Failed to clear display reserve bus feedback at %s.", self.path, exc_info=True)
                return
            _fsync_directory(self.path.parent)

    def record_reaction(
        self,
        *,
        topic_key: str,
        reaction: str,
        intensity: float,
        reason: str,
        now: datetime | None = None,
        source: str = "display_reserve_card",
        ttl_s: float | None = None,
    ) -> DisplayReserveBusFeedbackSignal:
        """Persist one new short-lived reserve-bus reaction hint."""

        effective_now = _normalize_datetime(now, default_now=True) or _utc_now()
        lifetime_s = _normalize_ttl_seconds(ttl_s, default=self.default_ttl_s)
        signal = DisplayReserveBusFeedbackSignal(
            schema_version=_SCHEMA_VERSION,
            source=source,
            requested_at=_format_timestamp(effective_now),
            expires_at=_format_timestamp(effective_now + timedelta(seconds=lifetime_s)),
            topic_key=topic_key,
            reaction=reaction,
            intensity=intensity,
            reason=reason,
        )
        self._validate_persistable_signal(signal)
        with _locked_file(self.lock_path):
            existing = self._load_unlocked(strict=True)
            if existing is not None and existing.requested_at_datetime() > signal.requested_at_datetime():
                _LOGGER.info(
                    "Ignoring stale display reserve bus feedback for %s because a newer signal already exists.",
                    signal.topic_key,
                )
                return existing
            return self._write_signal_unlocked(signal)


__all__ = ["DisplayReserveBusFeedbackSignal", "DisplayReserveBusFeedbackStore"]