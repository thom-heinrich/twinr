"""Persist and normalize short-lived display wake cues.

The display wake cue is a display-only artifact. It lets the authoritative
remote wake path request immediate visual feedback without mutating the runtime
status machine into a speculative listening state.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import json
import math
import os
from os import PathLike
from pathlib import Path
import stat
import tempfile
from types import ModuleType
from typing import Protocol

fcntl: ModuleType | None
try:
    import fcntl
except ImportError:  # pragma: no cover - Linux runtime expected.
    fcntl = None


_DEFAULT_WAKE_CUE_PATH = "artifacts/stores/ops/display_wake_cue.json"
_DEFAULT_WAKE_CUE_TTL_S = 4.5
_MAX_WAKE_CUE_BYTES = 4096
_MAX_SOURCE_LEN = 80
_MAX_KIND_LEN = 48
_MAX_TEXT_LEN = 160
_MAX_BACKEND_LEN = 64
_MAX_DETECTOR_LEN = 64


class _DisplayConfigLike(Protocol):
    """Describe the minimal config surface needed by the wake cue store."""

    project_root: str | PathLike[str]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _compact_text(value: object | None, *, max_len: int) -> str:
    if value is None:
        return ""
    text = "".join(ch if ch.isprintable() else " " for ch in str(value))
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 1].rstrip() + "…"


def _coerce_positive_seconds(value: object | None, *, default: float) -> float:
    candidate: object = default if value is None else value
    if not isinstance(candidate, (str, bytes, bytearray, int, float)):
        candidate = str(candidate)
    try:
        seconds = float(candidate)
    except (TypeError, ValueError):
        seconds = float(default)
    if not math.isfinite(seconds):
        seconds = float(default)
    return max(0.1, seconds)


def _normalize_timestamp(value: object | None, *, field: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class DisplayWakeCue:
    """Describe one short-lived visual wake cue for the display layer."""

    source: str
    kind: str = "speculative_wake"
    updated_at: str | None = None
    expires_at: str | None = None
    matched_phrase: str | None = None
    backend: str | None = None
    detector_label: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        source = _compact_text(self.source, max_len=_MAX_SOURCE_LEN)
        if not source:
            raise ValueError("source is required")
        kind = _compact_text(self.kind, max_len=_MAX_KIND_LEN)
        if kind != "speculative_wake":
            raise ValueError("kind must be 'speculative_wake'")
        matched_phrase = _compact_text(self.matched_phrase, max_len=_MAX_TEXT_LEN) or None
        backend = _compact_text(self.backend, max_len=_MAX_BACKEND_LEN) or None
        detector_label = _compact_text(self.detector_label, max_len=_MAX_DETECTOR_LEN) or None
        score = self.score
        if score is not None:
            score = float(score)
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError("score must be a finite ratio in [0.0, 1.0]")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "matched_phrase", matched_phrase)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "detector_label", detector_label)
        object.__setattr__(self, "score", score)
        if self.updated_at is not None:
            object.__setattr__(
                self,
                "updated_at",
                _format_timestamp(_normalize_timestamp(self.updated_at, field="updated_at")),
            )
        if self.expires_at is not None:
            object.__setattr__(
                self,
                "expires_at",
                _format_timestamp(_normalize_timestamp(self.expires_at, field="expires_at")),
            )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source": self.source,
            "kind": self.kind,
        }
        if self.updated_at is not None:
            payload["updated_at"] = self.updated_at
        if self.expires_at is not None:
            payload["expires_at"] = self.expires_at
        if self.matched_phrase is not None:
            payload["matched_phrase"] = self.matched_phrase
        if self.backend is not None:
            payload["backend"] = self.backend
        if self.detector_label is not None:
            payload["detector_label"] = self.detector_label
        if self.score is not None:
            payload["score"] = float(self.score)
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> "DisplayWakeCue":
        if not isinstance(payload, dict):
            raise ValueError("display wake cue payload must be an object")
        return cls(
            source=str(payload.get("source") or ""),
            kind=payload.get("kind", "speculative_wake"),
            updated_at=payload.get("updated_at"),
            expires_at=payload.get("expires_at"),
            matched_phrase=payload.get("matched_phrase"),
            backend=payload.get("backend"),
            detector_label=payload.get("detector_label"),
            score=payload.get("score"),
        )

    def is_active(self, *, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return _normalize_timestamp(self.expires_at, field="expires_at") > (now or _utc_now())

    def visual_signature(self) -> tuple[str]:
        """Return the visual identity that affects rendering."""

        return (self.kind,)

    def telemetry_signature(self) -> str:
        return self.kind


@dataclass(slots=True)
class DisplayWakeCueStore:
    """Read and write the short-lived display wake cue artifact."""

    path: Path
    default_ttl_s: float = _DEFAULT_WAKE_CUE_TTL_S
    max_file_bytes: int = _MAX_WAKE_CUE_BYTES

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self.default_ttl_s = _coerce_positive_seconds(
            self.default_ttl_s,
            default=_DEFAULT_WAKE_CUE_TTL_S,
        )
        self.max_file_bytes = max(512, int(self.max_file_bytes))

    @classmethod
    def from_config(cls, config: _DisplayConfigLike) -> "DisplayWakeCueStore":
        project_root = Path(config.project_root).expanduser().resolve()
        configured_path = Path(
            getattr(config, "display_wake_cue_path", _DEFAULT_WAKE_CUE_PATH) or _DEFAULT_WAKE_CUE_PATH
        )
        resolved_path = configured_path if configured_path.is_absolute() else project_root / configured_path
        return cls(
            path=resolved_path,
            default_ttl_s=getattr(config, "display_wake_cue_ttl_s", _DEFAULT_WAKE_CUE_TTL_S),
            max_file_bytes=getattr(config, "display_wake_cue_max_bytes", _MAX_WAKE_CUE_BYTES),
        )

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f"{self.path.name}.lock")

    @contextmanager
    def _lock(self, *, shared: bool) -> Iterator[None]:
        if not self.path.parent.exists():
            if shared:
                yield
                return
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if fcntl is None:  # pragma: no cover - Linux runtime expected.
            yield
            return
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self._lock_path, flags, 0o600)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise RuntimeError(f"Unsafe display wake cue lock path: {self._lock_path}") from exc
            raise
        try:
            fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _fsync_parent_dir(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        try:
            fd = os.open(self.path.parent, flags)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def load(self) -> DisplayWakeCue | None:
        if not self.path.parent.exists():
            return None
        with self._lock(shared=True):
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self.path, flags)
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENXIO}:
                    raise RuntimeError(f"Unsafe display wake cue path: {self.path}") from exc
                raise
            try:
                file_stat = os.fstat(fd)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise RuntimeError(f"display wake cue path is not a regular file: {self.path}")
                if file_stat.st_size > self.max_file_bytes:
                    raise ValueError(
                        f"display wake cue exceeds {self.max_file_bytes} bytes: {self.path}"
                    )
                raw = os.read(fd, self.max_file_bytes + 1)
                if len(raw) > self.max_file_bytes:
                    raise ValueError(
                        f"display wake cue exceeds {self.max_file_bytes} bytes while reading: {self.path}"
                    )
            finally:
                os.close(fd)
        payload = json.loads(raw)
        return DisplayWakeCue.from_dict(payload)

    def load_active(self, *, now: datetime | None = None) -> DisplayWakeCue | None:
        cue = self.load()
        if cue is None or not cue.is_active(now=now):
            return None
        return cue

    def save(
        self,
        cue: DisplayWakeCue,
        *,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayWakeCue:
        safe_now = (now or _utc_now()).astimezone(timezone.utc)
        ttl_s = _coerce_positive_seconds(hold_seconds, default=self.default_ttl_s)
        normalized = DisplayWakeCue(
            source=cue.source,
            kind=cue.kind,
            updated_at=_format_timestamp(safe_now),
            expires_at=_format_timestamp(safe_now + timedelta(seconds=ttl_s)),
            matched_phrase=cue.matched_phrase,
            backend=cue.backend,
            detector_label=cue.detector_label,
            score=cue.score,
        )
        encoded = json.dumps(
            normalized.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ValueError(
                f"Display wake cue payload exceeds {self.max_file_bytes} bytes after serialization."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._lock(shared=False):
            fd, tmp_name = tempfile.mkstemp(
                prefix=f"{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
                text=False,
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as tmp_file:
                    tmp_file.write(encoded)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                os.replace(tmp_name, self.path)
                self._fsync_parent_dir()
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise
        return normalized

    def clear(self) -> None:
        if not self.path.parent.exists():
            return
        with self._lock(shared=False):
            try:
                path_stat = self.path.lstat()
            except FileNotFoundError:
                return
            if stat.S_ISDIR(path_stat.st_mode):
                raise RuntimeError(f"Refusing to clear directory-backed display wake cue: {self.path}")
            self.path.unlink(missing_ok=True)
            self._fsync_parent_dir()


@dataclass(slots=True)
class DisplayWakeCueController:
    """Persist producer-facing display wake cues through the cue store."""

    store: DisplayWakeCueStore
    default_source: str = "external"

    @classmethod
    def from_config(
        cls,
        config: _DisplayConfigLike,
        *,
        default_source: str = "external",
    ) -> "DisplayWakeCueController":
        return cls(
            store=DisplayWakeCueStore.from_config(config),
            default_source=default_source,
        )

    def show_speculative_wake(
        self,
        *,
        source: str | None = None,
        matched_phrase: str | None = None,
        backend: str | None = None,
        detector_label: str | None = None,
        score: float | None = None,
        hold_seconds: float | None = None,
        now: datetime | None = None,
    ) -> DisplayWakeCue:
        return self.store.save(
            DisplayWakeCue(
                source=source or self.default_source,
                matched_phrase=matched_phrase,
                backend=backend,
                detector_label=detector_label,
                score=score,
            ),
            hold_seconds=hold_seconds,
            now=now,
        )

    def clear(self) -> None:
        self.store.clear()
