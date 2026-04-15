# CHANGELOG: 2026-03-28
# BUG-1: Fixed QR signature collisions; signature() now hashes the full QR payload instead of using only length + prefix.
# BUG-2: Fixed silent QR corruption; oversized qr_image_data_url values are no longer truncated into invalid base64.
# BUG-3: Fixed load() race with concurrent unlink/replace by reading via one file descriptor and using fstat() on that descriptor.
# BUG-4: Fixed non-finite TTL crashes; NaN/inf hold_seconds/default_ttl_s are now rejected instead of overflowing later.
# SEC-1: Writes are now atomic + fsynced + lock-serialized to survive power loss and concurrent writers on Raspberry Pi SD storage.
# SEC-2: Files/locks are created with private permissions and only safe regular files are read; QR payloads are restricted to bounded base64 raster images.
# IMP-1: Added an optional msgspec fast-path for schema-backed JSON decode/encode on constrained edge hardware.
# IMP-2: Added bounded file-size enforcement and compact JSON output to reduce memory spikes and write amplification.
# BUG-5: signature() now excludes updated_at/expires_at so cue lifetime refreshes do not
#        force semantically identical HDMI reserve rerenders on hdmi_wayland.

"""Persist active service-connect reserve cues for Twinr's HDMI info lane.

Service onboarding is stronger than ordinary ambient reserve content because
the user may need to scan a QR code or read a short status while they are in a
voice-triggered connect flow. This module keeps that right-lane payload
bounded, file-backed, and independent from the generic runtime snapshot.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import base64
import binascii
import errno
import hashlib
import json
import logging
import math
import os
from os import PathLike
from pathlib import Path
import stat
import tempfile
import time
from typing import BinaryIO, Protocol


_DEFAULT_SERVICE_CONNECT_TTL_S = 20.0
_DEFAULT_SERVICE_CONNECT_PATH = "artifacts/stores/ops/display_service_connect.json"
_DEFAULT_SERVICE_CONNECT_MAX_FILE_BYTES = 196_608
_DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES = 65_536
_DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS = 131_072
_DEFAULT_SERVICE_CONNECT_LOCK_TIMEOUT_S = 0.5
_ALLOWED_ACCENTS = frozenset({"neutral", "info", "success", "warm", "alert"})
_ALLOWED_QR_DATA_URL_PREFIXES = (
    "data:image/png;base64,",
    "data:image/webp;base64,",
    "data:image/jpeg;base64,",
    "data:image/gif;base64,",
)

_LOGGER = logging.getLogger(__name__)

try:
    import fcntl
except Exception:  # pragma: no cover - unavailable on non-Unix platforms
    fcntl = None

try:
    import msgspec
except Exception:  # pragma: no cover - optional frontier dependency
    msgspec = None


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the service-connect store."""

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

    if max_len < 1:
        raise ValueError("max_len must be positive")
    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _normalize_accent(value: object | None) -> str:
    """Normalize one optional visual accent token."""

    compact = _compact_text(value, max_len=24).lower().replace("-", "_").replace(" ", "_")
    if compact not in _ALLOWED_ACCENTS:
        return "info"
    return compact


def _coerce_positive_finite_seconds(
    value: object | None,
    *,
    default: float,
    field_name: str,
    minimum: float = 0.1,
) -> float:
    """Return one positive finite duration in seconds."""

    if value is None:
        return default
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number of seconds") from exc
    if not math.isfinite(seconds):
        raise ValueError(f"{field_name} must be a finite number of seconds")
    return max(minimum, seconds)


def _coerce_optional_int(
    value: object | None,
    *,
    default: int,
    minimum: int,
    maximum: int | None,
    field_name: str,
) -> int:
    """Parse one optional integer config value, falling back on invalid input."""

    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        _LOGGER.warning(
            "Invalid %s=%r for display service-connect store; using default %d.",
            field_name,
            value,
            default,
        )
        return default
    if parsed < minimum:
        _LOGGER.warning(
            "Invalid %s=%r for display service-connect store; using default %d.",
            field_name,
            value,
            default,
        )
        return default
    if maximum is not None and parsed > maximum:
        _LOGGER.warning(
            "Clamping %s=%r for display service-connect store to %d.",
            field_name,
            value,
            maximum,
        )
        return maximum
    return parsed


def _coerce_optional_seconds(
    value: object | None,
    *,
    default: float,
    minimum: float,
    field_name: str,
) -> float:
    """Parse one optional float duration config value, falling back on invalid input."""

    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _LOGGER.warning(
            "Invalid %s=%r for display service-connect store; using default %.3f.",
            field_name,
            value,
            default,
        )
        return default
    if not math.isfinite(parsed) or parsed < minimum:
        _LOGGER.warning(
            "Invalid %s=%r for display service-connect store; using default %.3f.",
            field_name,
            value,
            default,
        )
        return default
    return parsed


def _normalize_qr_image_data_url(
    value: object | None,
    *,
    max_chars: int,
    max_decoded_bytes: int,
    allow_invalid: bool,
) -> str | None:
    """Validate one QR image data URL without mutating its binary payload."""

    if value in (None, ""):
        return None
    text = "".join(ch for ch in str(value).strip() if not ch.isspace())
    if not text:
        return None
    if len(text) > max_chars:
        if allow_invalid:
            return None
        raise ValueError("qr_image_data_url exceeds maximum supported size")

    lowered = text.lower()
    matched_prefix = next((prefix for prefix in _ALLOWED_QR_DATA_URL_PREFIXES if lowered.startswith(prefix)), None)
    if matched_prefix is None:
        if allow_invalid:
            return None
        # BREAKING: only base64-encoded raster image data URLs are accepted now.
        raise ValueError("qr_image_data_url must be a base64-encoded raster image data URL")

    encoded = text[len(matched_prefix) :]
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        if allow_invalid:
            return None
        raise ValueError("qr_image_data_url is not valid base64 image data") from exc

    if len(decoded) > max_decoded_bytes:
        if allow_invalid:
            return None
        raise ValueError("qr_image_data_url decoded image exceeds maximum supported size")

    return text


def _digest_qr_signature(value: str | None) -> str | None:
    """Return one stable digest for QR payload comparison."""

    if not value:
        return None
    return hashlib.blake2s(value.encode("utf-8"), digest_size=16).hexdigest()


if msgspec is not None:

    class _WireCue(msgspec.Struct, kw_only=True):
        """Typed wire schema used when msgspec is available."""

        source: str = "service_connect"
        updated_at: str | None = None
        expires_at: str | None = None
        service_id: str = ""
        service_label: str = ""
        phase: str = ""
        summary: str = ""
        detail: str = ""
        qr_image_data_url: str | None = None
        accent: str = "info"

    def _wire_to_dict(value: _WireCue) -> dict[str, object]:
        return {
            "source": value.source,
            "updated_at": value.updated_at,
            "expires_at": value.expires_at,
            "service_id": value.service_id,
            "service_label": value.service_label,
            "phase": value.phase,
            "summary": value.summary,
            "detail": value.detail,
            "qr_image_data_url": value.qr_image_data_url,
            "accent": value.accent,
        }

    def _decode_payload_bytes(data: bytes) -> Mapping[str, object]:
        return _wire_to_dict(msgspec.json.decode(data, type=_WireCue))

    def _encode_payload_bytes(payload: Mapping[str, object]) -> bytes:
        wire = _WireCue(
            source=str(payload.get("source", "service_connect")),
            updated_at=payload.get("updated_at") if isinstance(payload.get("updated_at"), str) else None,
            expires_at=payload.get("expires_at") if isinstance(payload.get("expires_at"), str) else None,
            service_id=str(payload.get("service_id", "")),
            service_label=str(payload.get("service_label", "")),
            phase=str(payload.get("phase", "")),
            summary=str(payload.get("summary", "")),
            detail=str(payload.get("detail", "")),
            qr_image_data_url=(
                payload.get("qr_image_data_url") if isinstance(payload.get("qr_image_data_url"), str) else None
            ),
            accent=str(payload.get("accent", "info")),
        )
        return msgspec.json.encode(wire)

else:

    def _decode_payload_bytes(data: bytes) -> Mapping[str, object]:
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise TypeError("display service-connect payload is not a JSON object")
        return payload

    def _encode_payload_bytes(payload: Mapping[str, object]) -> bytes:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for the containing directory after replace()."""

    if os.name != "posix":
        return
    directory_fd: int | None = None
    try:
        directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(directory_fd)
    except OSError:
        _LOGGER.debug("Failed to fsync directory %s.", path, exc_info=True)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


@contextmanager
def _locked_file(lock_path: Path, *, timeout_s: float) -> Iterator[None]:
    """Acquire one sidecar advisory file lock, when supported."""

    if fcntl is None:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    deadline = time.monotonic() + max(0.0, timeout_s)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock {lock_path}") from exc
                time.sleep(0.01)
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _open_existing_regular_file(path: Path) -> tuple[BinaryIO, os.stat_result] | None:
    """Open one existing regular file safely for bounded reading."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError:
        _LOGGER.warning("Failed to open display service-connect cue from %s.", path, exc_info=True)
        return None

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            _LOGGER.warning("Ignoring non-regular display service-connect cue artifact at %s.", path)
            os.close(fd)
            return None
        return os.fdopen(fd, "rb"), st
    except Exception:
        os.close(fd)
        raise


@dataclass(frozen=True, slots=True)
class DisplayServiceConnectCue:
    """Describe one active service-connect reserve cue for the HDMI panel."""

    source: str = "service_connect"
    updated_at: str | None = None
    expires_at: str | None = None
    service_id: str = ""
    service_label: str = ""
    phase: str = ""
    summary: str = ""
    detail: str = ""
    qr_image_data_url: str | None = None
    accent: str = "info"
    _expires_at_dt: datetime | None = field(init=False, repr=False, compare=False)
    _qr_signature_digest: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _compact_text(self.source, max_len=80) or "service_connect")
        object.__setattr__(self, "service_id", _compact_text(self.service_id, max_len=48).casefold())
        object.__setattr__(self, "service_label", _compact_text(self.service_label, max_len=48))
        object.__setattr__(self, "phase", _compact_text(self.phase, max_len=48).lower())
        object.__setattr__(self, "summary", _compact_text(self.summary, max_len=96))
        object.__setattr__(self, "detail", _compact_text(self.detail, max_len=180))
        object.__setattr__(
            self,
            "qr_image_data_url",
            _normalize_qr_image_data_url(
                self.qr_image_data_url,
                max_chars=_DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS,
                max_decoded_bytes=_DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES,
                allow_invalid=False,
            ),
        )
        object.__setattr__(self, "accent", _normalize_accent(self.accent))

        updated_at_dt = _normalize_timestamp(self.updated_at)
        expires_at_dt = _normalize_timestamp(self.expires_at)
        # BREAKING: direct construction now validates ISO-8601 timestamps eagerly.
        if self.updated_at is not None and updated_at_dt is None:
            raise ValueError("updated_at must be an ISO-8601 timestamp")
        if self.expires_at is not None and expires_at_dt is None:
            raise ValueError("expires_at must be an ISO-8601 timestamp")

        object.__setattr__(self, "_expires_at_dt", expires_at_dt)
        object.__setattr__(self, "_qr_signature_digest", _digest_qr_signature(self.qr_image_data_url))

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        fallback_updated_at: datetime | None = None,
        default_ttl_s: float = _DEFAULT_SERVICE_CONNECT_TTL_S,
        max_qr_data_url_chars: int = _DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS,
        max_qr_image_bytes: int = _DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES,
    ) -> "DisplayServiceConnectCue":
        """Build one normalized service-connect cue from JSON-style data."""

        safe_now = (fallback_updated_at or _utc_now()).astimezone(timezone.utc)
        updated_at = _normalize_timestamp(payload.get("updated_at")) or safe_now
        expires_at = _normalize_timestamp(payload.get("expires_at"))
        ttl_s = _coerce_positive_finite_seconds(
            default_ttl_s,
            default=_DEFAULT_SERVICE_CONNECT_TTL_S,
            field_name="default_ttl_s",
        )
        if expires_at is None:
            expires_at = updated_at + timedelta(seconds=ttl_s)

        qr_image_data_url = _normalize_qr_image_data_url(
            payload.get("qr_image_data_url"),
            max_chars=max_qr_data_url_chars,
            max_decoded_bytes=max_qr_image_bytes,
            allow_invalid=True,
        )

        return cls(
            source=_compact_text(payload.get("source"), max_len=80) or "service_connect",
            updated_at=_format_timestamp(updated_at),
            expires_at=_format_timestamp(expires_at),
            service_id=_compact_text(payload.get("service_id"), max_len=48).casefold(),
            service_label=_compact_text(payload.get("service_label"), max_len=48),
            phase=_compact_text(payload.get("phase"), max_len=48).lower(),
            summary=_compact_text(payload.get("summary"), max_len=96),
            detail=_compact_text(payload.get("detail"), max_len=180),
            qr_image_data_url=qr_image_data_url,
            accent=_normalize_accent(payload.get("accent")),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the cue into a JSON-safe mapping."""

        return {
            "source": self.source,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "service_id": self.service_id,
            "service_label": self.service_label,
            "phase": self.phase,
            "summary": self.summary,
            "detail": self.detail,
            "qr_image_data_url": self.qr_image_data_url,
            "accent": self.accent,
        }

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return whether the cue should still affect the HDMI scene."""

        if self._expires_at_dt is None:
            return True
        return self._expires_at_dt >= (now or _utc_now()).astimezone(timezone.utc)

    def signature(self) -> tuple[object, ...]:
        """Return one stable render-signature fragment for this cue."""

        return (
            self.source,
            self.service_id,
            self.service_label,
            self.phase,
            self.summary,
            self.detail,
            self._qr_signature_digest,
            self.accent,
        )


@dataclass(slots=True)
class DisplayServiceConnectCueStore:
    """Read and write the optional HDMI service-connect cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_SERVICE_CONNECT_TTL_S
    max_file_bytes: int = _DEFAULT_SERVICE_CONNECT_MAX_FILE_BYTES
    max_qr_image_bytes: int = _DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES
    max_qr_data_url_chars: int = _DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS
    lock_timeout_s: float = _DEFAULT_SERVICE_CONNECT_LOCK_TIMEOUT_S

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayServiceConnectCueStore":
        """Resolve the service-connect cue path and limits from Twinr configuration."""

        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_service_connect_path", _DEFAULT_SERVICE_CONNECT_PATH)
            or _DEFAULT_SERVICE_CONNECT_PATH
        ).expanduser()
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path

        return cls(
            path=resolved_path,
            default_ttl_s=_coerce_optional_seconds(
                getattr(config, "display_service_connect_ttl_s", _DEFAULT_SERVICE_CONNECT_TTL_S),
                default=_DEFAULT_SERVICE_CONNECT_TTL_S,
                minimum=0.1,
                field_name="display_service_connect_ttl_s",
            ),
            max_file_bytes=_coerce_optional_int(
                getattr(config, "display_service_connect_max_file_bytes", _DEFAULT_SERVICE_CONNECT_MAX_FILE_BYTES),
                default=_DEFAULT_SERVICE_CONNECT_MAX_FILE_BYTES,
                minimum=1024,
                maximum=None,
                field_name="display_service_connect_max_file_bytes",
            ),
            max_qr_image_bytes=_coerce_optional_int(
                getattr(
                    config,
                    "display_service_connect_max_qr_image_bytes",
                    _DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES,
                ),
                default=_DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES,
                minimum=1024,
                maximum=_DEFAULT_SERVICE_CONNECT_MAX_QR_IMAGE_BYTES,
                field_name="display_service_connect_max_qr_image_bytes",
            ),
            max_qr_data_url_chars=_coerce_optional_int(
                getattr(
                    config,
                    "display_service_connect_max_qr_data_url_chars",
                    _DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS,
                ),
                default=_DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS,
                minimum=1024,
                maximum=_DEFAULT_SERVICE_CONNECT_MAX_QR_DATA_URL_CHARS,
                field_name="display_service_connect_max_qr_data_url_chars",
            ),
            lock_timeout_s=_coerce_optional_seconds(
                getattr(config, "display_service_connect_lock_timeout_s", _DEFAULT_SERVICE_CONNECT_LOCK_TIMEOUT_S),
                default=_DEFAULT_SERVICE_CONNECT_LOCK_TIMEOUT_S,
                minimum=0.0,
                field_name="display_service_connect_lock_timeout_s",
            ),
        )

    @property
    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> DisplayServiceConnectCue | None:
        """Load the current service-connect cue, if one exists and parses."""

        opened = _open_existing_regular_file(self.path)
        if opened is None:
            return None

        fh, st = opened
        with fh:
            try:
                raw = fh.read(self.max_file_bytes + 1)
            except Exception:
                _LOGGER.warning("Failed to read display service-connect cue from %s.", self.path, exc_info=True)
                return None

        if len(raw) > self.max_file_bytes:
            _LOGGER.warning(
                "Ignoring display service-connect cue at %s because it exceeds %d bytes.",
                self.path,
                self.max_file_bytes,
            )
            return None

        try:
            payload = _decode_payload_bytes(raw)
        except Exception:
            _LOGGER.warning("Failed to parse display service-connect cue from %s.", self.path, exc_info=True)
            return None

        fallback_updated_at = datetime.fromtimestamp(st.st_mtime_ns / 1_000_000_000, tz=timezone.utc)
        try:
            return DisplayServiceConnectCue.from_dict(
                payload,
                fallback_updated_at=fallback_updated_at,
                default_ttl_s=self.default_ttl_s,
                max_qr_data_url_chars=self.max_qr_data_url_chars,
                max_qr_image_bytes=self.max_qr_image_bytes,
            )
        except Exception:
            _LOGGER.warning("Ignoring invalid display service-connect cue payload at %s.", self.path, exc_info=True)
            return None

    def load_active(self, *, now: datetime | None = None) -> DisplayServiceConnectCue | None:
        """Load the current cue only when it is still active."""

        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayServiceConnectCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayServiceConnectCue:
        """Persist one service-connect cue with a bounded expiry."""

        written_at = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = _coerce_positive_finite_seconds(
            hold_seconds,
            default=self.default_ttl_s,
            field_name="hold_seconds",
        )
        normalized = DisplayServiceConnectCue(
            source=cue.source,
            updated_at=_format_timestamp(written_at),
            expires_at=_format_timestamp(written_at + timedelta(seconds=ttl_s)),
            service_id=cue.service_id,
            service_label=cue.service_label,
            phase=cue.phase,
            summary=cue.summary,
            detail=cue.detail,
            qr_image_data_url=cue.qr_image_data_url,
            accent=cue.accent,
        )

        payload_bytes = _encode_payload_bytes(normalized.to_dict()) + b"\n"
        if len(payload_bytes) > self.max_file_bytes:
            raise ValueError(
                f"display service-connect payload exceeds max_file_bytes={self.max_file_bytes}"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        with _locked_file(self._lock_path, timeout_s=self.lock_timeout_s):
            temp_fd: int | None = None
            temp_path_str = ""
            try:
                temp_fd, temp_path_str = tempfile.mkstemp(
                    prefix=f".{self.path.name}.",
                    suffix=".tmp",
                    dir=str(self.path.parent),
                    text=False,
                )
                if hasattr(os, "fchmod"):
                    os.fchmod(temp_fd, 0o600)
                with os.fdopen(temp_fd, "wb") as fh:
                    temp_fd = None
                    fh.write(payload_bytes)
                    fh.flush()
                    os.fsync(fh.fileno())

                os.replace(temp_path_str, self.path)
                _fsync_directory(self.path.parent)
            finally:
                if temp_fd is not None:
                    os.close(temp_fd)
                if temp_path_str:
                    try:
                        os.unlink(temp_path_str)
                    except FileNotFoundError:
                        pass

        return normalized

    def clear(self) -> None:
        """Remove the persisted cue artifact when it exists."""

        if not self.path.parent.exists():
            return
        try:
            with _locked_file(self._lock_path, timeout_s=self.lock_timeout_s):
                self.path.unlink()
        except FileNotFoundError:
            return


__all__ = [
    "DisplayServiceConnectCue",
    "DisplayServiceConnectCueStore",
]
