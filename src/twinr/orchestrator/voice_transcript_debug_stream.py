# CHANGELOG: 2026-03-29
# BUG-1: append_entry no longer lets filesystem/serialization failures break the live voice gateway; evidence persistence is now fail-open with internal health counters.
# BUG-2: fixed real append/trim race conditions that could corrupt or drop JSONL records under concurrent writers/processes.
# BUG-3: fixed unbounded sample/details serialization and brittle score coercion that could trigger crashes or oversized records on Raspberry Pi deployments.
# BUG-4: fixed relative-path resolution so transcript paths consistently resolve from the effective project root, including "~" expansion.
# SEC-1: transcript evidence files are now created with owner-only permissions by default and can reject symlink targets to reduce practical local data-exposure/overwrite risk.
# SEC-2: added optional on-device text redaction hooks for transcript evidence so operators can minimize stored PII/secrets when exact raw text retention is not required.
# IMP-1: upgraded the record schema toward 2026 observability practice with schema versioning, stable event names, resource metadata, correlation fields, and monotonic record IDs.
# IMP-2: added bounded-bytes retention, atomic trimming via temp-file replace, integrity chaining, and health snapshots so the JSONL stream is inspectable, bounded, and operationally robust.

"""Persist bounded textual voice-gateway transcript evidence.

The live transcript-first gateway on thh1986 must make its raw STT decisions
inspectable after the fact. This module appends one compact JSONL record per
transcribed voice decision window so operators can see exactly what the
gateway heard without persisting room audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import tempfile
import threading
import uuid

from twinr.agent.base_agent.config import TwinrConfig

try:  # Linux / Raspberry Pi deployment path.
    import fcntl
except Exception:  # pragma: no cover - non-Unix fallback
    fcntl = None


_DEFAULT_FILE_NAME = "voice_gateway_transcripts.jsonl"
_DEFAULT_MAX_LINES = 4096
_DEFAULT_MAX_BYTES = 8 * 1024 * 1024
_DEFAULT_TRIM_EVERY = 64
_DEFAULT_MAX_TEXT_LEN = 4096
_DEFAULT_MAX_ENTRY_BYTES = 16 * 1024
_DEFAULT_MAX_COLLECTION_ITEMS = 32
_DEFAULT_MAX_COLLECTION_DEPTH = 4
_DEFAULT_FILE_MODE = 0o600
_DEFAULT_DIR_MODE = 0o700
_DEFAULT_FSYNC_EVERY = 0
_DEFAULT_EVENT_NAME = "twinr.voice.gateway.transcript_window"
_DEFAULT_SCHEMA_VERSION = 2
_DEFAULT_SERVICE_NAME = "twinr.voice_gateway"
_SENTINEL_DEPTH = "<max_depth_exceeded>"
_SENTINEL_CYCLE = "<cycle_detected>"
_SENTINEL_TRUNCATED = "__truncated__"
_LOCK_SUFFIX = ".lock"

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.+-])",
    flags=re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")
_SECRET_RE = re.compile(
    r"(?i)\b(bearer|token|api[_ -]?key|secret|password|passcode|pin)\b"
    r"([^\S\r\n]*[:=][^\S\r\n]*|[^\S\r\n]+)(\S+)"
)

_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso_z(now: datetime | None = None) -> str:
    current = now or _utc_now()
    return current.strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_now_unix_nano(now: datetime | None = None) -> int:
    current = now or _utc_now()
    return int(current.timestamp() * 1_000_000_000)


def _bounded_text(value: object, *, limit: int = _DEFAULT_MAX_TEXT_LEN) -> str:
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _normalize_int(value: object, *, default: int, minimum: int) -> int:
    try:
        normalized = int(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _normalize_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_score(value: object) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return _bounded_text(value, limit=64)
    return normalized if math.isfinite(normalized) else _bounded_text(value, limit=64)


def _redact_text(
    value: str,
    *,
    redact_emails: bool,
    redact_phones: bool,
    redact_secrets: bool,
) -> str:
    text = value
    if redact_emails and text:
        text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    if redact_phones and text:
        text = _PHONE_RE.sub("[REDACTED_PHONE]", text)
    if redact_secrets and text:
        text = _SECRET_RE.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED_SECRET]",
            text,
        )
    return text


def _default_project_root(config: TwinrConfig | object) -> Path:
    """Prefer the configured project root and otherwise stay under the cwd.

    The live gateway may be started from a temporary env file outside the
    leading repo. For transcript evidence we still want the artifact anchored
    under the actual repo/runtime checkout, not under whatever fallback
    `runtime_state_path` happens to point to.
    """

    raw_project_root = str(getattr(config, "project_root", "") or "").strip()
    if raw_project_root and raw_project_root != ".":
        candidate = Path(raw_project_root).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve(strict=False)
        else:
            candidate = candidate.resolve(strict=False)
        return candidate
    return Path.cwd().resolve(strict=False)


@dataclass(frozen=True, slots=True)
class VoiceTranscriptDebugStreamConfig:
    """Store bounded transcript-debug stream settings."""

    path: Path
    enabled: bool = True
    max_lines: int = _DEFAULT_MAX_LINES
    max_bytes: int = _DEFAULT_MAX_BYTES
    trim_every: int = _DEFAULT_TRIM_EVERY
    max_text_len: int = _DEFAULT_MAX_TEXT_LEN
    max_entry_bytes: int = _DEFAULT_MAX_ENTRY_BYTES
    max_collection_items: int = _DEFAULT_MAX_COLLECTION_ITEMS
    max_collection_depth: int = _DEFAULT_MAX_COLLECTION_DEPTH
    fsync_every: int = _DEFAULT_FSYNC_EVERY
    secure_permissions: bool = True
    reject_symlinks: bool = True
    # BREAKING: symlinked transcript targets are rejected by default; set
    # `voice_orchestrator_transcript_debug_reject_symlinks=False` to retain
    # legacy symlink-based storage behavior.
    file_mode: int = _DEFAULT_FILE_MODE
    dir_mode: int = _DEFAULT_DIR_MODE
    enable_integrity_chain: bool = True
    integrity_key: str | None = None
    service_name: str = _DEFAULT_SERVICE_NAME
    redact_emails: bool = False
    redact_phones: bool = False
    redact_secrets: bool = False

    def __post_init__(self) -> None:
        bounded_entry_bytes = min(max(2048, self.max_entry_bytes), max(2048, self.max_bytes))
        object.__setattr__(self, "max_lines", max(1, self.max_lines))
        object.__setattr__(self, "max_bytes", max(2048, self.max_bytes))
        object.__setattr__(self, "trim_every", max(1, self.trim_every))
        object.__setattr__(self, "max_text_len", max(64, self.max_text_len))
        object.__setattr__(self, "max_entry_bytes", bounded_entry_bytes)
        object.__setattr__(self, "max_collection_items", max(1, self.max_collection_items))
        object.__setattr__(self, "max_collection_depth", max(1, self.max_collection_depth))
        object.__setattr__(self, "fsync_every", max(0, self.fsync_every))
        object.__setattr__(self, "file_mode", self.file_mode & 0o777)
        object.__setattr__(self, "dir_mode", self.dir_mode & 0o777)

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceTranscriptDebugStreamConfig":
        """Build one bounded transcript-debug config from Twinr runtime config."""

        project_root = _default_project_root(config)
        project_path = project_root / "artifacts" / "stores" / "ops" / _DEFAULT_FILE_NAME
        raw_path = getattr(config, "voice_orchestrator_transcript_debug_path", project_path)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (project_root / path).resolve(strict=False)
        else:
            path = path.resolve(strict=False)

        max_lines = _normalize_int(
            getattr(config, "voice_orchestrator_transcript_debug_max_lines", _DEFAULT_MAX_LINES),
            default=_DEFAULT_MAX_LINES,
            minimum=256,
        )
        max_bytes = _normalize_int(
            getattr(config, "voice_orchestrator_transcript_debug_max_bytes", _DEFAULT_MAX_BYTES),
            default=_DEFAULT_MAX_BYTES,
            minimum=256 * 1024,
        )
        max_entry_bytes = _normalize_int(
            getattr(config, "voice_orchestrator_transcript_debug_max_entry_bytes", _DEFAULT_MAX_ENTRY_BYTES),
            default=_DEFAULT_MAX_ENTRY_BYTES,
            minimum=2048,
        )
        if max_entry_bytes > max_bytes:
            max_entry_bytes = max_bytes

        return cls(
            path=path,
            enabled=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_enabled", True),
                default=True,
            ),
            max_lines=max_lines,
            max_bytes=max_bytes,
            trim_every=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_trim_every", _DEFAULT_TRIM_EVERY),
                default=_DEFAULT_TRIM_EVERY,
                minimum=1,
            ),
            max_text_len=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_max_text_len", _DEFAULT_MAX_TEXT_LEN),
                default=_DEFAULT_MAX_TEXT_LEN,
                minimum=64,
            ),
            max_entry_bytes=max_entry_bytes,
            max_collection_items=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_max_collection_items", _DEFAULT_MAX_COLLECTION_ITEMS),
                default=_DEFAULT_MAX_COLLECTION_ITEMS,
                minimum=4,
            ),
            max_collection_depth=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_max_collection_depth", _DEFAULT_MAX_COLLECTION_DEPTH),
                default=_DEFAULT_MAX_COLLECTION_DEPTH,
                minimum=1,
            ),
            fsync_every=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_fsync_every", _DEFAULT_FSYNC_EVERY),
                default=_DEFAULT_FSYNC_EVERY,
                minimum=0,
            ),
            secure_permissions=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_secure_permissions", True),
                default=True,
            ),
            reject_symlinks=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_reject_symlinks", True),
                default=True,
            ),
            file_mode=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_file_mode", _DEFAULT_FILE_MODE),
                default=_DEFAULT_FILE_MODE,
                minimum=0,
            )
            & 0o777,
            dir_mode=_normalize_int(
                getattr(config, "voice_orchestrator_transcript_debug_dir_mode", _DEFAULT_DIR_MODE),
                default=_DEFAULT_DIR_MODE,
                minimum=0,
            )
            & 0o777,
            enable_integrity_chain=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_enable_integrity_chain", True),
                default=True,
            ),
            integrity_key=_bounded_text(
                getattr(config, "voice_orchestrator_transcript_debug_integrity_key", None),
                limit=512,
            )
            or None,
            service_name=_bounded_text(
                getattr(config, "voice_orchestrator_transcript_debug_service_name", _DEFAULT_SERVICE_NAME),
                limit=128,
            )
            or _DEFAULT_SERVICE_NAME,
            redact_emails=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_redact_emails", False),
                default=False,
            ),
            redact_phones=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_redact_phones", False),
                default=False,
            ),
            redact_secrets=_normalize_bool(
                getattr(config, "voice_orchestrator_transcript_debug_redact_secrets", False),
                default=False,
            ),
        )


def _json_safe(
    value: object,
    *,
    text_limit: int,
    max_items: int,
    max_depth: int,
    redact_emails: bool,
    redact_phones: bool,
    redact_secrets: bool,
    depth: int = 0,
    seen: set[int] | None = None,
) -> object:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return _bounded_text(str(value), limit=text_limit)
    if isinstance(value, bytes):
        return _bounded_text(value.decode("utf-8", errors="replace"), limit=text_limit)
    if isinstance(value, str):
        return _bounded_text(
            _redact_text(
                value,
                redact_emails=redact_emails,
                redact_phones=redact_phones,
                redact_secrets=redact_secrets,
            ),
            limit=text_limit,
        )

    if depth >= max_depth:
        return _SENTINEL_DEPTH

    mutable = isinstance(value, (dict, list, tuple, set))
    object_id = id(value)
    if mutable:
        if seen is None:
            seen = set()
        if object_id in seen:
            return _SENTINEL_CYCLE
        seen.add(object_id)

    try:
        if isinstance(value, dict):
            result: dict[str, object] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= max_items:
                    result[_SENTINEL_TRUNCATED] = f"{index}+ items"
                    break
                safe_key = _bounded_text(key, limit=256)
                result[safe_key] = _json_safe(
                    item,
                    text_limit=text_limit,
                    max_items=max_items,
                    max_depth=max_depth,
                    redact_emails=redact_emails,
                    redact_phones=redact_phones,
                    redact_secrets=redact_secrets,
                    depth=depth + 1,
                    seen=seen,
                )
            return result

        if isinstance(value, (list, tuple, set)):
            result_list: list[object] = []
            for index, item in enumerate(value):
                if index >= max_items:
                    result_list.append(f"{_SENTINEL_TRUNCATED}:{index}+ items")
                    break
                result_list.append(
                    _json_safe(
                        item,
                        text_limit=text_limit,
                        max_items=max_items,
                        max_depth=max_depth,
                        redact_emails=redact_emails,
                        redact_phones=redact_phones,
                        redact_secrets=redact_secrets,
                        depth=depth + 1,
                        seen=seen,
                    )
                )
            return result_list
    finally:
        if mutable and seen is not None:
            seen.discard(object_id)

    return _bounded_text(
        _redact_text(
            repr(value),
            redact_emails=redact_emails,
            redact_phones=redact_phones,
            redact_secrets=redact_secrets,
        ),
        limit=text_limit,
    )


def _json_dumps_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_dumps_bytes(value: object) -> bytes:
    return (_json_dumps_compact(value) + "\n").encode("utf-8")


def _new_record_id() -> str:
    uuid7 = getattr(uuid, "uuid7", None)
    if callable(uuid7):
        return str(uuid7())  # pylint: disable=not-callable
    return str(uuid.uuid4())


def _get_path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve(strict=False))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown-host"


def _secure_mkdir(path: Path, *, mode: int) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=mode)


def _safe_stat_is_symlink(path: Path) -> bool:
    try:
        return path.is_symlink()
    except OSError:
        return False


def _os_open_flags(*, append: bool, reject_symlinks: bool) -> int:
    flags = os.O_WRONLY | os.O_CREAT
    if append:
        flags |= os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if reject_symlinks and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _safe_open_fd(
    path: Path,
    *,
    append: bool,
    reject_symlinks: bool,
    mode: int,
) -> int:
    if reject_symlinks and _safe_stat_is_symlink(path):
        raise OSError(f"Refusing to open symlinked transcript path: {path}")
    return os.open(str(path), _os_open_flags(append=append, reject_symlinks=reject_symlinks), mode)


def _lock_file(lock_fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)


def _unlock_file(lock_fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)


def _write_all_or_rollback(fd: int, payload: bytes) -> None:
    end_offset = os.lseek(fd, 0, os.SEEK_END)
    written = 0
    try:
        while written < len(payload):
            chunk = os.write(fd, payload[written:])
            if chunk <= 0:
                raise OSError("short write while appending transcript evidence")
            written += chunk
    except Exception:
        try:
            os.ftruncate(fd, end_offset)
        except OSError:
            pass
        raise


def _read_last_complete_line(path: Path) -> bytes | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end <= 0:
                return None
            scan = min(end, 65536)
            handle.seek(end - scan)
            chunk = handle.read(scan)
    except OSError:
        return None

    if not chunk:
        return None

    if chunk.endswith(b"\n"):
        trimmed = chunk[:-1]
    else:
        newline_index = chunk.rfind(b"\n")
        if newline_index < 0:
            return None
        trimmed = chunk[:newline_index]

    last_newline = trimmed.rfind(b"\n")
    if last_newline >= 0:
        return trimmed[last_newline + 1 :]
    if trimmed:
        return trimmed
    return None


def _digest_entry(entry_without_integrity: dict[str, object], *, key: str | None) -> tuple[str, str]:
    canonical = _json_dumps_compact(entry_without_integrity).encode("utf-8")
    if key:
        digest = hmac.new(key.encode("utf-8"), canonical, digestmod=hashlib.sha256).hexdigest()
        return "hmac-sha256", digest
    digest = hashlib.sha256(canonical).hexdigest()
    return "sha256", digest


class VoiceTranscriptDebugStream:
    """Append bounded transcript evidence for the live voice gateway."""

    def __init__(self, *, config: VoiceTranscriptDebugStreamConfig) -> None:
        self.config = config
        self._writes_since_trim = 0
        self._writes_since_fsync = 0
        self._write_failures = 0
        self._trim_failures = 0
        self._last_error: str | None = None
        self._last_write_at: str | None = None
        self._last_trim_at: str | None = None
        self._local_lock = _get_path_lock(config.path)
        self._resource = {
            "service.name": _bounded_text(config.service_name, limit=128) or _DEFAULT_SERVICE_NAME,
            "service.instance.id": f"{_safe_hostname()}:{os.getpid()}",
            "host.name": _safe_hostname(),
            "process.pid": os.getpid(),
            "python.version": platform.python_version(),
            "runtime": platform.python_implementation(),
        }

    @classmethod
    def from_config(cls, config: TwinrConfig | object) -> "VoiceTranscriptDebugStream":
        """Build one transcript-debug stream rooted under Twinr's ops store."""

        return cls(config=VoiceTranscriptDebugStreamConfig.from_config(config))

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def health_snapshot(self) -> dict[str, object]:
        """Expose bounded writer health for diagnostics without raising."""

        return {
            "enabled": self.config.enabled,
            "path": str(self.config.path),
            "write_failures": self._write_failures,
            "trim_failures": self._trim_failures,
            "last_error": self._last_error,
            "last_write_at": self._last_write_at,
            "last_trim_at": self._last_trim_at,
        }

    def append_entry(
        self,
        *,
        session_id: str | None,
        trace_id: str | None,
        state: str,
        backend: str,
        stage: str,
        outcome: str,
        transcript: str | None = None,
        matched_phrase: str | None = None,
        remaining_text: str | None = None,
        detector_label: str | None = None,
        score: float | None = None,
        sample: dict[str, object] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Append one bounded transcript-debug record.

        This writer is intentionally fail-open: transcript evidence must never
        take down the live voice gateway if the filesystem or serializer fails.
        """

        if not self.config.enabled:
            return

        now = _utc_now()
        write_failures_before = self._write_failures
        trim_failures_before = self._trim_failures
        try:
            entry = self._build_entry(
                now=now,
                session_id=session_id,
                trace_id=trace_id,
                state=state,
                backend=backend,
                stage=stage,
                outcome=outcome,
                transcript=transcript,
                matched_phrase=matched_phrase,
                remaining_text=remaining_text,
                detector_label=detector_label,
                score=score,
                sample=sample,
                details=details,
            )
            self._append_serialized_entry(entry)
            self._last_write_at = _utc_now_iso_z(now)
            if (
                self._write_failures == write_failures_before
                and self._trim_failures == trim_failures_before
            ):
                self._last_error = None
        except Exception as exc:  # pragma: no cover - fail-open path
            self._write_failures += 1
            self._last_error = _bounded_text(f"{type(exc).__name__}: {exc}", limit=512)

    def _build_entry(
        self,
        *,
        now: datetime,
        session_id: str | None,
        trace_id: str | None,
        state: str,
        backend: str,
        stage: str,
        outcome: str,
        transcript: str | None,
        matched_phrase: str | None,
        remaining_text: str | None,
        detector_label: str | None,
        score: float | None,
        sample: dict[str, object] | None,
        details: dict[str, object] | None,
    ) -> dict[str, object]:
        transcript_text = self._safe_optional_text(transcript)
        matched_phrase_text = self._safe_text(matched_phrase, limit=256) if matched_phrase else None
        remaining_text_value = self._safe_optional_text(remaining_text)
        detector_label_text = self._safe_text(detector_label, limit=128) if detector_label else None

        safe_sample = self._safe_json(sample or {})
        safe_details = self._safe_json(details or {})
        safe_score = _normalize_score(score)

        entry: dict[str, object] = {
            "schema_version": _DEFAULT_SCHEMA_VERSION,
            "event_name": _DEFAULT_EVENT_NAME,
            "created_at": _utc_now_iso_z(now),
            "timestamp_unix_nano": _utc_now_unix_nano(now),
            "observed_timestamp_unix_nano": _utc_now_unix_nano(now),
            "record_id": _new_record_id(),
            "session_id": self._safe_text(session_id, limit=128) if session_id else None,
            "trace_id": self._safe_text(trace_id, limit=128) if trace_id else None,
            "state": self._safe_text(state, limit=64) or "unknown",
            "backend": self._safe_text(backend, limit=64) or "unknown",
            "stage": self._safe_text(stage, limit=96) or "unknown",
            "outcome": self._safe_text(outcome, limit=128) or "unknown",
            "transcript": transcript_text,
            "matched_phrase": matched_phrase_text,
            "remaining_text": remaining_text_value,
            "detector_label": detector_label_text,
            "score": safe_score,
            "sample": safe_sample,
            "details": safe_details,
            "resource": dict(self._resource),
            "attributes": {
                "state": self._safe_text(state, limit=64) or "unknown",
                "backend": self._safe_text(backend, limit=64) or "unknown",
                "stage": self._safe_text(stage, limit=96) or "unknown",
                "outcome": self._safe_text(outcome, limit=128) or "unknown",
                "session.id": self._safe_text(session_id, limit=128) if session_id else None,
                "trace.id": self._safe_text(trace_id, limit=128) if trace_id else None,
                "voice.detector_label": detector_label_text,
            },
            "body": {
                "transcript": transcript_text,
                "matched_phrase": matched_phrase_text,
                "remaining_text": remaining_text_value,
                "score": safe_score,
            },
            "size_hints": {
                "transcript_chars": len(transcript_text) if transcript_text else 0,
                "remaining_text_chars": len(remaining_text_value) if remaining_text_value else 0,
            },
        }

        if self.config.enable_integrity_chain:
            entry["integrity"] = None

        return self._shrink_entry_to_budget(entry)

    def _safe_text(self, value: object, *, limit: int | None = None) -> str:
        return _bounded_text(
            _redact_text(
                "" if value is None else str(value),
                redact_emails=self.config.redact_emails,
                redact_phones=self.config.redact_phones,
                redact_secrets=self.config.redact_secrets,
            ),
            limit=limit or self.config.max_text_len,
        )

    def _safe_optional_text(self, value: object, *, limit: int | None = None) -> str | None:
        if value is None:
            return None
        text = self._safe_text(value, limit=limit)
        return text if text != "" else ""

    def _safe_json(self, value: object) -> object:
        return _json_safe(
            value,
            text_limit=self.config.max_text_len,
            max_items=self.config.max_collection_items,
            max_depth=self.config.max_collection_depth,
            redact_emails=self.config.redact_emails,
            redact_phones=self.config.redact_phones,
            redact_secrets=self.config.redact_secrets,
        )

    def _shrink_entry_to_budget(self, entry: dict[str, object]) -> dict[str, object]:
        attempts = 0
        while True:
            payload = _json_dumps_bytes(entry)
            if len(payload) <= self.config.max_entry_bytes:
                return entry
            attempts += 1
            if attempts == 1:
                entry["details"] = {"summary": "details elided to honor max_entry_bytes"}
                continue
            if attempts == 2:
                entry["sample"] = {"summary": "sample elided to honor max_entry_bytes"}
                continue
            if attempts == 3:
                entry["remaining_text"] = (
                    self._safe_text(
                        entry.get("remaining_text"),
                        limit=max(256, self.config.max_text_len // 2),
                    )
                    or None
                )
                entry["body"] = {
                    "transcript": entry.get("transcript"),
                    "matched_phrase": entry.get("matched_phrase"),
                    "remaining_text": entry.get("remaining_text"),
                    "score": entry.get("score"),
                }
                continue
            if attempts == 4:
                entry["transcript"] = (
                    self._safe_text(
                        entry.get("transcript"),
                        limit=max(256, self.config.max_text_len // 2),
                    )
                    or None
                )
                entry["matched_phrase"] = self._safe_text(entry.get("matched_phrase"), limit=128) or None
                entry["body"] = {
                    "transcript": entry.get("transcript"),
                    "matched_phrase": entry.get("matched_phrase"),
                    "remaining_text": entry.get("remaining_text"),
                    "score": entry.get("score"),
                }
                continue
            entry["details"] = {"summary": "record force-compacted"}
            entry["sample"] = {"summary": "record force-compacted"}
            entry["remaining_text"] = self._safe_text(entry.get("remaining_text"), limit=256) or None
            entry["transcript"] = self._safe_text(entry.get("transcript"), limit=256) or None
            entry["matched_phrase"] = self._safe_text(entry.get("matched_phrase"), limit=96) or None
            entry["body"] = {
                "transcript": entry.get("transcript"),
                "matched_phrase": entry.get("matched_phrase"),
                "remaining_text": entry.get("remaining_text"),
                "score": entry.get("score"),
            }
            return entry

    def _append_serialized_entry(self, entry: dict[str, object]) -> None:
        with self._local_lock:
            self._ensure_parent_dir()
            lock_fd = self._open_lock_fd()
            try:
                _lock_file(lock_fd)
                prev_digest = self._read_previous_digest() if self.config.enable_integrity_chain else None
                finalized = self._finalize_integrity(entry, prev_digest=prev_digest)
                payload = _json_dumps_bytes(finalized)
                fd = _safe_open_fd(
                    self.config.path,
                    append=True,
                    reject_symlinks=self.config.reject_symlinks,
                    mode=self.config.file_mode,
                )
                try:
                    if self.config.secure_permissions and hasattr(os, "fchmod"):
                        os.fchmod(fd, self.config.file_mode)
                    _write_all_or_rollback(fd, payload)
                    self._writes_since_trim += 1
                    self._writes_since_fsync += 1
                    size_after = os.fstat(fd).st_size
                    if self.config.fsync_every > 0 and self._writes_since_fsync >= self.config.fsync_every:
                        os.fsync(fd)
                        self._writes_since_fsync = 0
                finally:
                    os.close(fd)

                if self._should_trim(size_after):
                    self._trim_locked()
            finally:
                try:
                    _unlock_file(lock_fd)
                finally:
                    os.close(lock_fd)

    def _ensure_parent_dir(self) -> None:
        _secure_mkdir(self.config.path.parent, mode=self.config.dir_mode)

    def _open_lock_fd(self) -> int:
        lock_path = self.config.path.with_name(self.config.path.name + _LOCK_SUFFIX)
        return _safe_open_fd(
            lock_path,
            append=False,
            reject_symlinks=self.config.reject_symlinks,
            mode=self.config.file_mode,
        )

    def _read_previous_digest(self) -> str | None:
        last_line = _read_last_complete_line(self.config.path)
        if not last_line:
            return None
        try:
            record = json.loads(last_line.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return None
        integrity = record.get("integrity")
        if not isinstance(integrity, dict):
            return None
        digest = integrity.get("digest")
        return digest if isinstance(digest, str) and digest else None

    def _finalize_integrity(self, entry: dict[str, object], *, prev_digest: str | None) -> dict[str, object]:
        if not self.config.enable_integrity_chain:
            return entry
        entry_without_integrity = dict(entry)
        entry_without_integrity["integrity"] = {
            "algorithm": None,
            "prev_digest": prev_digest,
            "digest": None,
        }
        algorithm, digest = _digest_entry(entry_without_integrity, key=self.config.integrity_key)
        finalized = dict(entry)
        finalized["integrity"] = {
            "algorithm": algorithm,
            "prev_digest": prev_digest,
            "digest": digest,
        }
        return finalized

    def _should_trim(self, size_after: int) -> bool:
        if size_after > self.config.max_bytes:
            return True
        return self._writes_since_trim >= self.config.trim_every

    def _trim_locked(self) -> None:
        """Keep only the newest bounded slice of transcript-debug records."""

        self._writes_since_trim = 0
        if not self.config.path.is_file():
            return

        try:
            data = self.config.path.read_bytes()
        except OSError as exc:
            self._trim_failures += 1
            self._last_error = _bounded_text(f"{type(exc).__name__}: {exc}", limit=512)
            return

        if not data:
            return

        complete = data.endswith(b"\n")
        lines = data.splitlines()
        if not complete and lines:
            lines = lines[:-1]

        kept = self._select_kept_lines(lines)
        if self.config.enable_integrity_chain:
            kept = self._rechain_lines(kept)
        new_bytes = b"".join(line + b"\n" for line in kept)
        if new_bytes == data and len(kept) == len(lines):
            return

        tmp_fd = -1
        tmp_name = ""
        try:
            tmp_fd, tmp_name = tempfile.mkstemp(
                prefix=self.config.path.name + ".",
                suffix=".tmp",
                dir=str(self.config.path.parent),
                text=False,
            )
            if self.config.secure_permissions and hasattr(os, "fchmod"):
                os.fchmod(tmp_fd, self.config.file_mode)
            if new_bytes:
                _write_all_or_rollback(tmp_fd, new_bytes)
            if self.config.fsync_every > 0:
                os.fsync(tmp_fd)
            os.close(tmp_fd)
            tmp_fd = -1
            os.replace(tmp_name, self.config.path)
            self._last_trim_at = _utc_now_iso_z()
        except Exception as exc:  # pragma: no cover - rare filesystem failures
            self._trim_failures += 1
            self._last_error = _bounded_text(f"{type(exc).__name__}: {exc}", limit=512)
            if tmp_fd >= 0:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _rechain_lines(self, lines: list[bytes]) -> list[bytes]:
        if not lines:
            return lines

        rechained: list[bytes] = []
        prev_digest: str | None = None
        for line in lines:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                rechained.append(line)
                continue
            if not isinstance(record, dict):
                rechained.append(line)
                continue
            if "integrity" not in record:
                rechained.append(line)
                continue
            finalized = self._finalize_integrity(record, prev_digest=prev_digest)
            encoded = _json_dumps_bytes(finalized).rstrip(b"\n")
            rechained.append(encoded)
            integrity = finalized.get("integrity")
            if isinstance(integrity, dict):
                digest = integrity.get("digest")
                prev_digest = digest if isinstance(digest, str) and digest else prev_digest
        return rechained

    def _select_kept_lines(self, lines: list[bytes]) -> list[bytes]:
        if not lines:
            return []

        kept_reversed: list[bytes] = []
        total_bytes = 0
        for line in reversed(lines):
            line_size = len(line) + 1
            if kept_reversed and len(kept_reversed) >= self.config.max_lines:
                break
            if kept_reversed and total_bytes + line_size > self.config.max_bytes:
                break
            if not kept_reversed and line_size > self.config.max_bytes:
                kept_reversed.append(line)
                break
            kept_reversed.append(line)
            total_bytes += line_size
        kept_reversed.reverse()
        return kept_reversed


__all__ = [
    "VoiceTranscriptDebugStream",
    "VoiceTranscriptDebugStreamConfig",
]
