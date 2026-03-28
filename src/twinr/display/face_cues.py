from __future__ import annotations

# CHANGELOG: 2026-03-28
# BUG-1: Fixed a real multi-writer race in save(); the old fixed temp filename could
#        raise FileNotFoundError, lose updates, or expose transient invalid JSON.
# BUG-2: Fixed load()/clear() races against concurrent writers/removers; the old code
#        could still crash after a successful read due to a later stat()/unlink race.
# BUG-3: Fixed expiry drift caused by wall-clock jumps by persisting a monotonic
#        deadline alongside ISO timestamps and using it when the current boot matches.
# BUG-4: Fixed save(hold_seconds=...) not reliably extending/re-timing an existing cue;
#        hold_seconds now always refreshes the persisted deadline from the save time.
# SEC-1: Removed the predictable shared temp file and replaced it with secure same-dir
#        mkstemp writes + atomic replace, closing a practical symlink/clobber avenue.
# SEC-2: Reject symlink-backed cue artifacts on load and replace them on save, and cap
#        on-disk payload size to keep malicious or accidental large files from harming the Pi.
# IMP-1: Added inter-process advisory locking (flock on Linux/Unix) plus in-process
#        thread locking so the display loop and external triggers can coordinate safely.
# IMP-2: Added durable writes (flush + fsync + directory fsync where available).
# IMP-3: Added optional msgspec JSON decode acceleration when available, while keeping
#        the public dataclass API and the legacy flat JSON payload format compatible.
# IMP-4: Added more robust timestamp/TTL parsing (ISO-8601 or Unix epoch seconds) and
#        preserved drop-in compatibility for older flat payloads via a reserved _meta block.
# BREAKING: save(..., hold_seconds=...) now refreshes updated_at/expires_at from the save time
#           so the persisted deadline matches the caller's requested hold duration.
# BREAKING: the persisted artifact now inherits mkstemp owner-only permissions by default
#           (typically 0600). Cross-UID readers should switch to the same service user or a
#           dedicated group-owned path strategy.

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import errno
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import tempfile
import threading
import time
from typing import Protocol

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - not expected on Raspberry Pi / Linux
    fcntl = None  # type: ignore[assignment]

try:
    import msgspec  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional fast-path only
    msgspec = None  # type: ignore[assignment]


_DEFAULT_FACE_CUE_TTL_S = 4.0
_DEFAULT_FACE_CUE_PATH = "artifacts/stores/ops/display_face_cue.json"
_DEFAULT_FACE_CUE_LOCK_SUFFIX = ".lock"
_DEFAULT_FACE_CUE_LOCK_MODE = 0o600
_MAX_FACE_CUE_BYTES = 16 * 1024
_MAX_GAZE_AXIS = 3
_MAX_HEAD_AXIS = 2
_META_KEY = "_meta"
_META_SCHEMA_VERSION_KEY = "schema_version"
_META_BOOT_ID_KEY = "boot_id"
_META_EXPIRES_MONOTONIC_NS_KEY = "expires_monotonic_ns"
_META_WRITTEN_AT_MONOTONIC_NS_KEY = "written_at_monotonic_ns"
_STORAGE_SCHEMA_VERSION = 2
_MOUTH_ALIASES = {
    "concern": "sad",
    "line": "neutral",
}
_ALLOWED_MOUTHS = frozenset(
    {
        "neutral",
        "smile",
        "sad",
        "thinking",
        "pursed",
        "scrunched",
        "open",
        "speak",
    }
)
_BROW_ALIASES = {
    "concern": "inward_tilt",
    "flat": "straight",
    "focus": "inward_tilt",
}
_ALLOWED_BROWS = frozenset(
    {
        "straight",
        "inward_tilt",
        "outward_tilt",
        "roof",
        "raised",
        "soft",
    }
)

_LOGGER = logging.getLogger(__name__)


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the face cue store."""

    project_root: str | PathLike[str]


@lru_cache(maxsize=1)
def _boot_id() -> str | None:
    """Return the current Linux boot id when available."""

    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _utc_now() -> datetime:
    """Return the current UTC wall clock."""

    return datetime.now(timezone.utc)


def _monotonic_ns() -> int:
    """Return the current process-monotonic clock reading."""

    return time.monotonic_ns()


def _normalize_timestamp(value: object | None) -> datetime | None:
    """Parse one optional timestamp into an aware UTC datetime.

    Accepts ISO-8601 text (including a trailing Z) and Unix epoch seconds.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            epoch_seconds = float(text)
        except ValueError:
            return None
        if not math.isfinite(epoch_seconds):
            return None
        try:
            return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    """Serialize one aware timestamp as ISO-8601 text."""

    return value.astimezone(timezone.utc).isoformat()


def _normalize_duration_seconds(value: object | None, *, fallback: float) -> float:
    """Normalize one duration into a small positive finite number of seconds."""

    candidate = fallback if value is None else value
    try:
        parsed = float(candidate)
    except (TypeError, ValueError):
        parsed = float(fallback)
    if not math.isfinite(parsed):
        parsed = float(fallback)
    return max(0.1, parsed)


def _clamp_axis(value: object | None, *, maximum: int) -> int:
    """Normalize one signed cue-axis value into the supported range."""

    numeric: object
    numeric = int(value) if isinstance(value, bool) else (0 if value is None else value)
    try:
        parsed_float = float(numeric)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(parsed_float):
        return 0
    parsed = int(round(parsed_float))
    return max(-maximum, min(maximum, parsed))


def _normalize_style(
    value: object | None,
    *,
    allowed: frozenset[str],
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Normalize one optional face-style label against an explicit allow-list."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return None
    if aliases is not None:
        text = aliases.get(text, text)
    if text not in allowed:
        return None
    return text


def _normalize_optional_bool(value: object | None) -> bool | None:
    """Normalize one optional boolean face flag."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        if float(value) == 1:
            return True
        if float(value) == 0:
            return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_source(value: object | None) -> str:
    """Normalize one optional cue source label."""

    text = str(value or "").strip()
    if not text:
        return "external"
    text = " ".join(text.split())
    return text[:128] or "external"


def _normalize_optional_int(value: object | None) -> int | None:
    """Normalize one optional integer field."""

    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _decode_json_bytes(payload: bytes) -> object:
    """Decode one small JSON document, preferring msgspec when installed."""

    if msgspec is not None:
        return msgspec.json.decode(payload)  # type: ignore[union-attr]
    return json.loads(payload.decode("utf-8"))


def _encode_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Encode one small JSON document."""

    # Keep the serialized output stable for diffs and debugging.
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _extract_storage_payload(payload: Mapping[str, object]) -> tuple[dict[str, object], Mapping[str, object]]:
    """Split a stored payload into the cue body and internal metadata."""

    cue_payload: dict[str, object]
    meta_payload: Mapping[str, object] = {}

    cue_block = payload.get("cue")
    if isinstance(cue_block, Mapping):
        cue_payload = dict(cue_block)
        raw_meta = payload.get(_META_KEY) or payload.get("meta")
        if isinstance(raw_meta, Mapping):
            meta_payload = raw_meta
        return cue_payload, meta_payload

    cue_payload = dict(payload)
    raw_meta = cue_payload.pop(_META_KEY, None)
    cue_payload.pop("meta", None)
    if isinstance(raw_meta, Mapping):
        meta_payload = raw_meta
    return cue_payload, meta_payload


def _build_storage_payload(cue: "DisplayFaceCue") -> dict[str, object]:
    """Build the on-disk JSON payload for one cue."""

    payload = cue.to_dict()
    meta_payload = {
        _META_SCHEMA_VERSION_KEY: _STORAGE_SCHEMA_VERSION,
        _META_BOOT_ID_KEY: cue._boot_id,
        _META_EXPIRES_MONOTONIC_NS_KEY: cue._expires_monotonic_ns,
        _META_WRITTEN_AT_MONOTONIC_NS_KEY: cue._written_at_monotonic_ns,
    }
    payload[_META_KEY] = {key: value for key, value in meta_payload.items() if value is not None}
    return payload


def _safe_directory_fsync(path: Path) -> None:
    """Best-effort fsync of one directory path after a metadata change."""

    try:
        dir_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        return
    finally:
        os.close(dir_fd)


def _read_small_file_bytes(path: Path) -> bytes:
    """Read one bounded-size file without racing Path.exists()/stat()."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    fd = os.open(path, flags)
    try:
        stat_result = os.fstat(fd)
        if stat_result.st_size > _MAX_FACE_CUE_BYTES:
            raise ValueError(f"display face cue exceeds {_MAX_FACE_CUE_BYTES} bytes")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(_MAX_FACE_CUE_BYTES + 1)
        if len(data) > _MAX_FACE_CUE_BYTES:
            raise ValueError(f"display face cue exceeds {_MAX_FACE_CUE_BYTES} bytes")
        return data
    finally:
        os.close(fd)


@dataclass(frozen=True, slots=True)
class DisplayFaceCue:
    """Describe one optional external face-expression cue."""

    source: str = "external"
    updated_at: str | None = None
    expires_at: str | None = None
    gaze_x: int = 0
    gaze_y: int = 0
    head_dx: int = 0
    head_dy: int = 0
    mouth: str | None = None
    brows: str | None = None
    blink: bool | None = None
    _boot_id: str | None = field(default=None, repr=False, compare=False)
    _expires_monotonic_ns: int | None = field(default=None, repr=False, compare=False)
    _written_at_monotonic_ns: int | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize direct constructor calls into the canonical cue vocabulary."""

        object.__setattr__(self, "source", _normalize_source(self.source))
        object.__setattr__(self, "gaze_x", _clamp_axis(self.gaze_x, maximum=_MAX_GAZE_AXIS))
        object.__setattr__(self, "gaze_y", _clamp_axis(self.gaze_y, maximum=_MAX_GAZE_AXIS))
        object.__setattr__(self, "head_dx", _clamp_axis(self.head_dx, maximum=_MAX_HEAD_AXIS))
        object.__setattr__(self, "head_dy", _clamp_axis(self.head_dy, maximum=_MAX_HEAD_AXIS))
        object.__setattr__(self, "mouth", _normalize_style(self.mouth, allowed=_ALLOWED_MOUTHS, aliases=_MOUTH_ALIASES))
        object.__setattr__(self, "brows", _normalize_style(self.brows, allowed=_ALLOWED_BROWS, aliases=_BROW_ALIASES))
        object.__setattr__(self, "blink", _normalize_optional_bool(self.blink))
        object.__setattr__(self, "_boot_id", str(self._boot_id).strip() or None)
        object.__setattr__(self, "_expires_monotonic_ns", _normalize_optional_int(self._expires_monotonic_ns))
        object.__setattr__(self, "_written_at_monotonic_ns", _normalize_optional_int(self._written_at_monotonic_ns))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_FACE_CUE_TTL_S,
        meta: Mapping[str, object] | None = None,
    ) -> "DisplayFaceCue":
        """Build one normalized face cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        safe_ttl_s = _normalize_duration_seconds(default_ttl_s, fallback=_DEFAULT_FACE_CUE_TTL_S)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=safe_ttl_s)
        source = _normalize_source(payload.get("source", "external"))
        safe_meta = meta if isinstance(meta, Mapping) else {}
        return cls(
            source=source,
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            gaze_x=_clamp_axis(payload.get("gaze_x"), maximum=_MAX_GAZE_AXIS),
            gaze_y=_clamp_axis(payload.get("gaze_y"), maximum=_MAX_GAZE_AXIS),
            head_dx=_clamp_axis(payload.get("head_dx"), maximum=_MAX_HEAD_AXIS),
            head_dy=_clamp_axis(payload.get("head_dy"), maximum=_MAX_HEAD_AXIS),
            mouth=_normalize_style(payload.get("mouth"), allowed=_ALLOWED_MOUTHS, aliases=_MOUTH_ALIASES),
            brows=_normalize_style(payload.get("brows"), allowed=_ALLOWED_BROWS, aliases=_BROW_ALIASES),
            blink=_normalize_optional_bool(payload.get("blink")),
            _boot_id=str(safe_meta.get(_META_BOOT_ID_KEY) or "").strip() or None,
            _expires_monotonic_ns=_normalize_optional_int(safe_meta.get(_META_EXPIRES_MONOTONIC_NS_KEY)),
            _written_at_monotonic_ns=_normalize_optional_int(safe_meta.get(_META_WRITTEN_AT_MONOTONIC_NS_KEY)),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping.

        Internal transport metadata is deliberately excluded so callers continue to
        see the legacy flat cue schema.
        """

        return {
            "source": self.source,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "gaze_x": self.gaze_x,
            "gaze_y": self.gaze_y,
            "head_dx": self.head_dx,
            "head_dy": self.head_dy,
            "mouth": self.mouth,
            "brows": self.brows,
            "blink": self.blink,
        }

    def is_active(self, *, now: datetime | None = None, monotonic_ns: int | None = None) -> bool:
        """Return whether the cue should still influence the HDMI face."""

        current_boot_id = _boot_id()
        current_monotonic_ns = _monotonic_ns() if monotonic_ns is None else monotonic_ns
        if (
            now is None
            and monotonic_ns is None
            and self._boot_id is not None
            and current_boot_id is not None
            and self._boot_id == current_boot_id
            and self._expires_monotonic_ns is not None
        ):
            return self._expires_monotonic_ns >= current_monotonic_ns
        expires_at = _normalize_timestamp(self.expires_at)
        if expires_at is None:
            return True
        return expires_at >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return a stable render-signature fragment for this cue."""

        return (
            self.source,
            self.updated_at,
            self.expires_at,
            self.gaze_x,
            self.gaze_y,
            self.head_dx,
            self.head_dy,
            self.mouth,
            self.brows,
            self.blink,
        )


@dataclass(slots=True)
class DisplayFaceCueStore:
    """Read and write the optional face-cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_FACE_CUE_TTL_S
    _thread_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize constructor inputs."""

        self.path = Path(self.path).expanduser()
        self.default_ttl_s = _normalize_duration_seconds(self.default_ttl_s, fallback=_DEFAULT_FACE_CUE_TTL_S)

    @property
    def lock_path(self) -> Path:
        """Return the sidecar lock file used for cross-process coordination."""

        return self.path.parent / f".{self.path.name}{_DEFAULT_FACE_CUE_LOCK_SUFFIX}"

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayFaceCueStore":
        """Resolve the face-cue path from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve(strict=False)
        configured_path = Path(
            getattr(config, "display_face_cue_path", _DEFAULT_FACE_CUE_PATH) or _DEFAULT_FACE_CUE_PATH
        ).expanduser()
        resolved_path = configured_path if configured_path.is_absolute() else (project_root / configured_path)
        return cls(
            path=resolved_path.resolve(strict=False),
            default_ttl_s=_normalize_duration_seconds(
                getattr(config, "display_face_cue_ttl_s", _DEFAULT_FACE_CUE_TTL_S),
                fallback=_DEFAULT_FACE_CUE_TTL_S,
            ),
        )

    def _open_lock_fd(self) -> int:
        """Open the sidecar lock file safely, healing a symlinked lock path once."""

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if nofollow:
            flags |= nofollow
        for _ in range(2):
            try:
                return os.open(self.lock_path, flags, _DEFAULT_FACE_CUE_LOCK_MODE)
            except OSError as exc:
                if exc.errno != errno.ELOOP:
                    raise
                try:
                    if self.lock_path.is_symlink():
                        self.lock_path.unlink()
                        continue
                except FileNotFoundError:
                    continue
                raise
        raise RuntimeError(f"Unable to open display face cue lock file: {self.lock_path}")

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        """Serialize access across threads and cooperating processes."""

        with self._thread_lock:
            if fcntl is None:  # pragma: no cover - not expected on Raspberry Pi / Linux
                yield
                return
            lock_fd = self._open_lock_fd()
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                yield
            finally:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)

    def load(self) -> DisplayFaceCue | None:
        """Load the current face cue, if one exists and parses."""

        with self._locked(exclusive=False):
            try:
                raw_bytes = _read_small_file_bytes(self.path)
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    _LOGGER.warning("Ignoring symlink-backed display face cue at %s.", self.path)
                    return None
                _LOGGER.warning("Failed to read display face cue from %s.", self.path, exc_info=True)
                return None
            except ValueError:
                _LOGGER.warning("Ignoring oversized display face cue payload at %s.", self.path)
                return None

            if not raw_bytes.strip():
                return None

            try:
                payload = _decode_json_bytes(raw_bytes)
            except Exception:
                _LOGGER.warning("Failed to decode display face cue from %s.", self.path, exc_info=True)
                return None
            if not isinstance(payload, Mapping):
                _LOGGER.warning("Ignoring invalid display face cue payload at %s because it is not an object.", self.path)
                return None

            try:
                stat_result = self.path.stat()
            except FileNotFoundError:
                return None
            fallback_updated_at = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc)
            cue_payload, meta_payload = _extract_storage_payload(payload)
            try:
                return DisplayFaceCue.from_dict(
                    cue_payload,
                    fallback_updated_at=fallback_updated_at,
                    default_ttl_s=self.default_ttl_s,
                    meta=meta_payload,
                )
            except Exception:
                _LOGGER.warning("Ignoring invalid display face cue payload at %s.", self.path, exc_info=True)
                return None

    def load_active(self, *, now: datetime | None = None) -> DisplayFaceCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayFaceCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayFaceCue:
        """Persist one normalized face cue atomically.

        BREAKING: when hold_seconds is provided, save() now refreshes the persisted
        updated_at/expires_at from the save time so callers get the hold they asked for.
        """

        effective_now = (now or _utc_now()).astimezone(timezone.utc)
        effective_monotonic_ns = _monotonic_ns()
        effective_ttl_s = _normalize_duration_seconds(
            self.default_ttl_s if hold_seconds is None else hold_seconds,
            fallback=self.default_ttl_s,
        )

        parsed_updated_at = _normalize_timestamp(cue.updated_at)
        parsed_expires_at = _normalize_timestamp(cue.expires_at)
        if hold_seconds is not None:
            updated_at_dt = effective_now
            expires_at_dt = effective_now + timedelta(seconds=effective_ttl_s)
        else:
            updated_at_dt = parsed_updated_at or effective_now
            expires_at_dt = parsed_expires_at or (updated_at_dt + timedelta(seconds=effective_ttl_s))

        remaining_seconds = max(0.0, (expires_at_dt - effective_now).total_seconds())
        normalized = DisplayFaceCue(
            source=cue.source,
            updated_at=_format_timestamp(updated_at_dt),
            expires_at=_format_timestamp(expires_at_dt),
            gaze_x=cue.gaze_x,
            gaze_y=cue.gaze_y,
            head_dx=cue.head_dx,
            head_dy=cue.head_dy,
            mouth=cue.mouth,
            brows=cue.brows,
            blink=cue.blink,
            _boot_id=_boot_id(),
            _written_at_monotonic_ns=effective_monotonic_ns,
            _expires_monotonic_ns=effective_monotonic_ns + int(round(remaining_seconds * 1_000_000_000)),
        )
        payload = _build_storage_payload(normalized)
        encoded = _encode_json_bytes(payload)
        if len(encoded) > _MAX_FACE_CUE_BYTES:
            raise ValueError(f"display face cue exceeds {_MAX_FACE_CUE_BYTES} bytes")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(exclusive=True):
            try:
                if self.path.is_symlink():
                    self.path.unlink()
            except FileNotFoundError:
                pass

            # BREAKING: mkstemp creates an owner-only file (typically 0600), which is
            # safer for deployed artifacts but may require coordination if readers were
            # previously running under a different UID.
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=str(self.path.parent),
                text=False,
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb", closefd=False) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.close(fd)
                os.replace(tmp_path, self.path)
                _safe_directory_fsync(self.path.parent)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    _LOGGER.debug("Failed to clean up temporary display face cue %s.", tmp_path, exc_info=True)
                raise
        return normalized

    def clear(self) -> None:
        """Remove the current face cue artifact if one exists."""

        with self._locked(exclusive=True):
            try:
                self.path.unlink(missing_ok=True)
            except TypeError:  # pragma: no cover - Python < 3.8
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    return
            _safe_directory_fsync(self.path.parent)
