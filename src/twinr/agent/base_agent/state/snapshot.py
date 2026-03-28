# CHANGELOG: 2026-03-27
# BUG-1: save() now round-trips persisted RuntimeSnapshot* entries and ISO timestamps instead of crashing on valid string created_at values.
# BUG-2: Falsey values like 0 and False in metadata, sources, and open_loops are no longer silently dropped during normalization.
# BUG-3: Snapshot reads and writes are now bounded by size and collection limits to prevent Pi-4 RAM, disk, and SD-card wear blowups.
# BUG-4: save() again accepts both object-backed runtime models and mapping-backed payloads, and restores the long-standing cross-service 0644 file-mode default.
# SEC-1: Snapshot reads now use O_NOFOLLOW to avoid symlink-race file traversal.
# SEC-2: The on-disk snapshot is now a versioned envelope with SHA-256 payload verification, so corrupted-but-parseable files are rejected and backup recovery works.
# IMP-1: Added optional msgspec/orjson fast-paths for decoding and hashing while keeping the on-disk JSON envelope human-readable.
# IMP-2: Added schema_version, revision, and optional optimistic concurrency via expected_revision for safer multi-service writes.
# IMP-3: Added bounded normalization/truncation for texts, turns, ledger items, and search results to keep snapshots compact and predictable.

"""Persist and restore normalized runtime snapshots for the base Twinr agent.

This module defines the snapshot schema shared by runtime, display, web, and
ops code and provides a file-backed store that reads malformed data
defensively, verifies payload integrity, and keeps snapshot I/O bounded for
Raspberry Pi deployments. Import the dataclasses when consuming persisted
runtime state and use ``RuntimeSnapshotStore`` for all snapshot I/O.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import math
import os
import stat
import tempfile
from typing import SupportsFloat, SupportsIndex, SupportsInt, cast

from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry


_DEFAULT_STATUS = "waiting"
_BACKUP_SUFFIX = ".bak"
_LOCK_SUFFIX = ".lock"

# BREAKING: Snapshot files are now written as a versioned integrity-checked envelope.
_CURRENT_SCHEMA_VERSION = 2
_CURRENT_FORMAT = "twinr.runtime_snapshot"

# Cross-service readers still rely on world-readable snapshots by default.
_DEFAULT_FILE_MODE = 0o644
_DEFAULT_LOCK_FILE_MODE = 0o666
_DEFAULT_MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024

_MAX_STATUS_CHARS = 64
_MAX_ROLE_CHARS = 32
_MAX_KIND_CHARS = 64
_MAX_ID_CHARS = 256
_MAX_NAME_CHARS = 256
_MAX_SOURCE_LABEL_CHARS = 128
_MAX_REASON_CHARS = 1024
_MAX_TOPIC_CHARS = 1024
_MAX_LOOP_CHARS = 512
_MAX_SOURCE_TEXT_CHARS = 2048
_MAX_METADATA_KEY_CHARS = 128
_MAX_METADATA_VALUE_CHARS = 1024

_MAX_TRANSCRIPT_CHARS = 65_536
_MAX_RESPONSE_CHARS = 65_536
_MAX_ERROR_CHARS = 16_384
_MAX_TURN_CONTENT_CHARS = 12_288
_MAX_LEDGER_CONTENT_CHARS = 8_192
_MAX_SEARCH_QUESTION_CHARS = 4_096
_MAX_SEARCH_ANSWER_CHARS = 16_384
_MAX_SEARCH_CONTEXT_CHARS = 256

_MAX_MEMORY_TURNS = 256
_MAX_MEMORY_RAW_TAIL = 128
_MAX_MEMORY_LEDGER = 256
_MAX_MEMORY_SEARCH_RESULTS = 128
_MAX_OPEN_LOOPS = 64
_MAX_SOURCES_PER_SEARCH_ENTRY = 32
_MAX_METADATA_ITEMS = 64

_IntLike = str | bytes | bytearray | SupportsInt | SupportsIndex
_FloatLike = str | bytes | bytearray | SupportsFloat | SupportsIndex


try:  # Optional frontier fast-path.
    import msgspec  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - dependency is optional
    msgspec = None

try:  # Optional fast JSON fallback when msgspec is unavailable.
    import orjson  # type: ignore[import-not-found]  # pylint: disable=import-error
except Exception:  # pragma: no cover - dependency is optional
    orjson = None


if msgspec is not None:  # pragma: no branch
    class _MsgspecEnvelope(msgspec.Struct):
        schema_version: int
        format: str
        revision: int
        payload_sha256: str
        payload: dict[str, object]

    _MSGSPEC_JSON_ENCODER = msgspec.json.Encoder()
    _MSGSPEC_JSON_DECODER = msgspec.json.Decoder()
    _MSGSPEC_ENVELOPE_DECODER = msgspec.json.Decoder(type=_MsgspecEnvelope)
else:  # pragma: no cover - simple placeholders for type checkers
    _MSGSPEC_JSON_ENCODER = None
    _MSGSPEC_JSON_DECODER = None
    _MSGSPEC_ENVELOPE_DECODER = None


class RuntimeSnapshotStoreError(RuntimeError):
    """Raised when the runtime snapshot store cannot safely read or write state."""


def _utcnow() -> datetime:
    """Return the current UTC time as an aware ``datetime``."""

    return datetime.now(timezone.utc)


def _refresh_fd_mode_best_effort(fd: int, mode: int) -> None:
    """Best-effort chmod for shared lock files that may be owned by another user."""

    try:
        if stat.S_IMODE(os.fstat(fd).st_mode) == mode:
            return
    except OSError:
        return
    try:
        os.fchmod(fd, mode)
    except PermissionError:
        return


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotTurn:
    """Represent one persisted conversation turn in the runtime snapshot."""

    role: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotLedgerItem:
    """Represent one persisted ledger entry in the runtime snapshot."""

    kind: str
    content: str
    created_at: str
    source: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotSearchEntry:
    """Represent one persisted verified-search memory entry."""

    question: str
    answer: str
    sources: tuple[str, ...]
    created_at: str
    location_hint: str | None = None
    date_context: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotMemoryState:
    """Represent the structured memory-state section of a snapshot."""

    active_topic: str | None = None
    last_user_goal: str | None = None
    pending_printable: str | None = None
    last_search_summary: str | None = None
    open_loops: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Represent the normalized runtime snapshot payload."""

    status: str = _DEFAULT_STATUS
    printing_active: bool = False
    last_transcript: str | None = None
    last_response: str | None = None
    updated_at: str | None = None
    error_message: str | None = None
    user_voice_status: str | None = None
    user_voice_confidence: float | None = None
    user_voice_checked_at: str | None = None
    user_voice_user_id: str | None = None
    user_voice_user_display_name: str | None = None
    user_voice_match_source: str | None = None
    voice_quiet_until_utc: str | None = None
    voice_quiet_reason: str | None = None
    memory_turns: tuple[RuntimeSnapshotTurn, ...] = ()
    memory_raw_tail: tuple[RuntimeSnapshotTurn, ...] = ()
    memory_ledger: tuple[RuntimeSnapshotLedgerItem, ...] = ()
    memory_search_results: tuple[RuntimeSnapshotSearchEntry, ...] = ()
    memory_state: RuntimeSnapshotMemoryState = field(default_factory=RuntimeSnapshotMemoryState)
    schema_version: int = _CURRENT_SCHEMA_VERSION
    revision: int = 0

    @property
    def memory_count(self) -> int:
        return len(self.memory_turns)


class RuntimeSnapshotStore:
    """Load and save runtime snapshots with locking, atomic writes, and recovery."""

    def __init__(
        self,
        path: str | Path,
        *,
        file_mode: int = _DEFAULT_FILE_MODE,
        max_snapshot_bytes: int = _DEFAULT_MAX_SNAPSHOT_BYTES,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.name or self.path.name in {".", ".."}:
            raise RuntimeSnapshotStoreError("Runtime snapshot path must point to a file.")
        self.file_mode = _coerce_file_mode(file_mode)
        self.max_snapshot_bytes = max(1024, int(max_snapshot_bytes))
        self._backup_path = self.path.with_name(f"{self.path.name}{_BACKUP_SUFFIX}")
        self._lock_path = self.path.with_name(f".{self.path.name}{_LOCK_SUFFIX}")

    @property
    def lock_path(self) -> Path:
        """Return the canonical cross-process lock file for this snapshot store."""

        return self._lock_path

    def load(self) -> RuntimeSnapshot:
        if not self.path.parent.exists():
            return RuntimeSnapshot()

        with self._locked(shared=True):
            self._validate_targets()
            snapshot = self._load_best_snapshot_unlocked()
            if snapshot is not None:
                return snapshot
        return RuntimeSnapshot()

    def save(
        self,
        *,
        status: str,
        printing_active: bool = False,
        memory_turns: tuple[ConversationTurn, ...],
        memory_raw_tail: tuple[ConversationTurn, ...] | None = None,
        memory_ledger: tuple[MemoryLedgerItem, ...] | None = None,
        memory_search_results: tuple[SearchMemoryEntry, ...] | None = None,
        memory_state: MemoryState | None = None,
        last_transcript: str | None,
        last_response: str | None,
        error_message: str | None = None,
        user_voice_status: str | None = None,
        user_voice_confidence: float | None = None,
        user_voice_checked_at: str | None = None,
        user_voice_user_id: str | None = None,
        user_voice_user_display_name: str | None = None,
        user_voice_match_source: str | None = None,
        voice_quiet_until_utc: str | None = None,
        voice_quiet_reason: str | None = None,
        expected_revision: int | None = None,
    ) -> RuntimeSnapshot:
        """Normalize and persist one runtime snapshot payload.

        ``expected_revision`` is optional. When supplied, save will fail if the
        on-disk snapshot revision changed since the caller last loaded it.
        """

        self._ensure_parent_directory()

        with self._locked(shared=False):
            self._validate_targets()
            current_snapshot = self._load_best_snapshot_unlocked()
            current_revision = current_snapshot.revision if current_snapshot is not None else 0

            if expected_revision is not None and expected_revision != current_revision:
                raise RuntimeSnapshotStoreError(
                    f"Runtime snapshot revision mismatch: expected {expected_revision}, found {current_revision}."
                )

            sanitized_confidence = _coerce_confidence(user_voice_confidence)

            snapshot = RuntimeSnapshot(
                status=_trimmed_str(status, max_chars=_MAX_STATUS_CHARS) or _DEFAULT_STATUS,
                printing_active=_normalize_printing_active(status=status, printing_active=printing_active),
                last_transcript=_coerce_optional_text(last_transcript, max_chars=_MAX_TRANSCRIPT_CHARS),
                last_response=_coerce_optional_text(last_response, max_chars=_MAX_RESPONSE_CHARS),
                updated_at=_utcnow().isoformat(),
                error_message=_coerce_optional_text(error_message, max_chars=_MAX_ERROR_CHARS),
                user_voice_status=_trimmed_str(user_voice_status, max_chars=_MAX_STATUS_CHARS),
                user_voice_confidence=sanitized_confidence,
                user_voice_checked_at=_coerce_optional_datetime_string(user_voice_checked_at),
                user_voice_user_id=_trimmed_str(user_voice_user_id, max_chars=_MAX_ID_CHARS),
                user_voice_user_display_name=_trimmed_str(
                    user_voice_user_display_name,
                    max_chars=_MAX_NAME_CHARS,
                ),
                user_voice_match_source=_trimmed_str(
                    user_voice_match_source,
                    max_chars=_MAX_SOURCE_LABEL_CHARS,
                ),
                voice_quiet_until_utc=_coerce_optional_datetime_string(voice_quiet_until_utc),
                voice_quiet_reason=_coerce_optional_text(voice_quiet_reason, max_chars=_MAX_REASON_CHARS),
                memory_turns=_normalize_runtime_turns(memory_turns, field_name="memory_turns"),
                memory_raw_tail=_normalize_runtime_turns(
                    memory_raw_tail or (),
                    field_name="memory_raw_tail",
                    max_items=_MAX_MEMORY_RAW_TAIL,
                ),
                memory_ledger=_normalize_runtime_ledger(memory_ledger or ()),
                memory_search_results=_normalize_runtime_search_results(memory_search_results or ()),
                memory_state=_normalize_memory_state(memory_state),
                schema_version=_CURRENT_SCHEMA_VERSION,
                revision=current_revision + 1,
            )

            payload = _snapshot_to_payload(snapshot)
            envelope = _build_envelope(snapshot, payload)
            envelope_bytes = _serialize_json_bytes(envelope)
            if len(envelope_bytes) > self.max_snapshot_bytes:
                raise RuntimeSnapshotStoreError(
                    "Runtime snapshot exceeds the configured size limit after normalization."
                )

            self._atomic_write_bytes(self.path, envelope_bytes)
            self._write_backup_best_effort(envelope_bytes)

        return snapshot

    def _load_best_snapshot_unlocked(self) -> RuntimeSnapshot | None:
        candidates: list[tuple[int, datetime, int, RuntimeSnapshot]] = []
        for source_rank, candidate in enumerate((self.path, self._backup_path), start=1):
            if not candidate.exists():
                continue
            snapshot = self._load_from_path(candidate)
            if snapshot is None:
                continue
            updated_at = _parse_datetime_like(snapshot.updated_at) or datetime.min.replace(tzinfo=timezone.utc)
            candidates.append((snapshot.revision, updated_at, -source_rank, snapshot))

        if not candidates:
            return None
        return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]

    def _load_from_path(self, candidate: Path) -> RuntimeSnapshot | None:
        try:
            payload_bytes = self._read_file_bytes_limited(candidate)
        except RuntimeSnapshotStoreError:
            return None

        data = self._deserialize_bytes(payload_bytes)
        if data is None:
            return None

        if msgspec is not None and isinstance(data, _MsgspecEnvelope):
            payload = data.payload
            if not self._is_valid_envelope(
                schema_version=data.schema_version,
                format_name=data.format,
                revision=data.revision,
                payload_sha256=data.payload_sha256,
                payload=payload,
            ):
                return None
            return _runtime_snapshot_from_mapping(
                payload,
                schema_version=_coerce_non_negative_int(data.schema_version, default=_CURRENT_SCHEMA_VERSION),
                revision=_coerce_non_negative_int(data.revision, default=0),
            )

        if not isinstance(data, Mapping):
            return None

        envelope_payload = data.get("payload")
        if isinstance(envelope_payload, Mapping):
            schema_version = _coerce_non_negative_int(data.get("schema_version"), default=_CURRENT_SCHEMA_VERSION)
            revision = _coerce_non_negative_int(data.get("revision"), default=0)
            format_name = _trimmed_str(data.get("format"), max_chars=128)
            payload_sha256 = _trimmed_str(data.get("payload_sha256"), max_chars=128)
            if not self._is_valid_envelope(
                schema_version=schema_version,
                format_name=format_name,
                revision=revision,
                payload_sha256=payload_sha256,
                payload=envelope_payload,
            ):
                return None
            return _runtime_snapshot_from_mapping(
                envelope_payload,
                schema_version=schema_version,
                revision=revision,
            )

        return _runtime_snapshot_from_mapping(data, schema_version=1, revision=0)

    def _is_valid_envelope(
        self,
        *,
        schema_version: int,
        format_name: str | None,
        revision: int,
        payload_sha256: str | None,
        payload: Mapping[str, object],
    ) -> bool:
        if schema_version < 2:
            return False
        if format_name != _CURRENT_FORMAT:
            return False
        if revision < 0 or not payload_sha256:
            return False
        return payload_sha256 == _payload_digest(payload)

    def _deserialize_bytes(self, payload_bytes: bytes) -> object | None:
        if msgspec is not None:
            try:
                return _MSGSPEC_ENVELOPE_DECODER.decode(payload_bytes)
            except Exception:
                pass
            try:
                return _MSGSPEC_JSON_DECODER.decode(payload_bytes)
            except Exception:
                return None

        if orjson is not None:
            try:
                return orjson.loads(payload_bytes)
            except Exception:
                return None

        try:
            return json.loads(payload_bytes)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _read_file_bytes_limited(self, candidate: Path) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(candidate, flags)
        except OSError as exc:
            raise RuntimeSnapshotStoreError(f"Unable to open runtime snapshot file: {candidate}") from exc

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeSnapshotStoreError(f"Refusing non-regular snapshot file: {candidate}")
            if file_stat.st_size > self.max_snapshot_bytes:
                raise RuntimeSnapshotStoreError(
                    f"Refusing oversized runtime snapshot file (> {self.max_snapshot_bytes} bytes): {candidate}"
                )
            with os.fdopen(fd, "rb", closefd=False) as handle:
                data = handle.read(self.max_snapshot_bytes + 1)
            if len(data) > self.max_snapshot_bytes:
                raise RuntimeSnapshotStoreError(
                    f"Refusing oversized runtime snapshot file (> {self.max_snapshot_bytes} bytes): {candidate}"
                )
            return data
        finally:
            os.close(fd)

    def _ensure_parent_directory(self) -> None:
        self._validate_parent_chain(self.path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeSnapshotStoreError("Unable to create the runtime snapshot directory.") from exc

    def _validate_targets(self) -> None:
        for target in (self.path, self._backup_path, self._lock_path):
            self._validate_parent_chain(target)
            self._validate_target_path(target)

    def _validate_target_path(self, target: Path) -> None:
        try:
            if target.is_symlink():
                raise RuntimeSnapshotStoreError(f"Refusing symlink target for runtime snapshot store: {target}")
            if target.exists() and not target.is_file():
                raise RuntimeSnapshotStoreError(
                    f"Refusing non-file target for runtime snapshot store: {target}"
                )
        except OSError as exc:
            raise RuntimeSnapshotStoreError(
                f"Unable to validate runtime snapshot target: {target}"
            ) from exc

    def _validate_parent_chain(self, target: Path) -> None:
        for parent in (target.parent, *target.parent.parents):
            try:
                if parent.is_symlink():
                    raise RuntimeSnapshotStoreError(
                        f"Refusing symlink parent for runtime snapshot store: {parent}"
                    )
                if parent.exists() and not parent.is_dir():
                    raise RuntimeSnapshotStoreError(
                        f"Runtime snapshot parent is not a directory: {parent}"
                    )
            except OSError as exc:
                raise RuntimeSnapshotStoreError(
                    f"Unable to validate runtime snapshot parent: {parent}"
                ) from exc

    @contextmanager
    def _locked(self, *, shared: bool) -> Iterator[None]:
        lock_handle = None
        try:
            self._validate_parent_chain(self._lock_path)
            self._validate_target_path(self._lock_path)
            lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            lock_fd = os.open(self._lock_path, lock_flags, 0o600)
            _refresh_fd_mode_best_effort(lock_fd, _DEFAULT_LOCK_FILE_MODE)
            lock_handle = os.fdopen(lock_fd, "a+", encoding="utf-8")
            lock_mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(lock_handle.fileno(), lock_mode)
            yield
        except OSError as exc:
            raise RuntimeSnapshotStoreError("Unable to lock the runtime snapshot store.") from exc
        finally:
            if lock_handle is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                lock_handle.close()

    def _atomic_write_bytes(self, target: Path, payload_bytes: bytes) -> None:
        temp_file_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload_bytes)
                handle.flush()
                os.fchmod(handle.fileno(), self.file_mode)
                os.fsync(handle.fileno())
                temp_file_path = Path(handle.name)

            os.replace(temp_file_path, target)
            os.chmod(target, self.file_mode)
            temp_file_path = None
            self._fsync_directory(target.parent)
        except OSError as exc:
            raise RuntimeSnapshotStoreError(
                f"Unable to write the runtime snapshot atomically: {target}"
            ) from exc
        finally:
            if temp_file_path is not None:
                try:
                    temp_file_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _write_backup_best_effort(self, payload_bytes: bytes) -> None:
        try:
            self._atomic_write_bytes(self._backup_path, payload_bytes)
        except RuntimeSnapshotStoreError:
            return

    def _fsync_directory(self, directory: Path) -> None:
        try:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
            directory_fd = os.open(directory, directory_flags)
        except OSError as exc:
            raise RuntimeSnapshotStoreError(
                f"Unable to open runtime snapshot directory for fsync: {directory}"
            ) from exc
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise RuntimeSnapshotStoreError(
                f"Unable to fsync runtime snapshot directory: {directory}"
            ) from exc
        finally:
            os.close(directory_fd)


def _coerce_file_mode(value: object) -> int:
    try:
        mode = int(cast(_IntLike, value)) & 0o777
    except (TypeError, ValueError):
        raise RuntimeSnapshotStoreError("Runtime snapshot file_mode must be an integer POSIX mode.") from None
    if mode == 0:
        raise RuntimeSnapshotStoreError("Runtime snapshot file_mode must grant at least one permission bit.")
    return mode


def _payload_digest(payload: Mapping[str, object]) -> str:
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeSnapshotStoreError("Unable to canonicalize the runtime snapshot payload.") from exc
    return text.encode("utf-8")


def _serialize_json_bytes(value: object) -> bytes:
    """Serialize the snapshot envelope as readable JSON for cross-service inspection."""

    if orjson is not None:
        try:
            return orjson.dumps(
                value,
                option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE,
            )
        except Exception as exc:
            raise RuntimeSnapshotStoreError("Unable to serialize the runtime snapshot.") from exc

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise RuntimeSnapshotStoreError("Unable to serialize the runtime snapshot.") from exc


def _build_envelope(snapshot: RuntimeSnapshot, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": _CURRENT_SCHEMA_VERSION,
        "format": _CURRENT_FORMAT,
        "revision": snapshot.revision,
        "payload_sha256": _payload_digest(payload),
        "payload": payload,
    }


def _snapshot_to_payload(snapshot: RuntimeSnapshot) -> dict[str, object]:
    return {
        "status": snapshot.status,
        "printing_active": snapshot.printing_active,
        "last_transcript": snapshot.last_transcript,
        "last_response": snapshot.last_response,
        "updated_at": snapshot.updated_at,
        "error_message": snapshot.error_message,
        "user_voice_status": snapshot.user_voice_status,
        "user_voice_confidence": snapshot.user_voice_confidence,
        "user_voice_checked_at": snapshot.user_voice_checked_at,
        "user_voice_user_id": snapshot.user_voice_user_id,
        "user_voice_user_display_name": snapshot.user_voice_user_display_name,
        "user_voice_match_source": snapshot.user_voice_match_source,
        "voice_quiet_until_utc": snapshot.voice_quiet_until_utc,
        "voice_quiet_reason": snapshot.voice_quiet_reason,
        "memory_turns": [
            {
                "role": turn.role,
                "content": turn.content,
                "created_at": turn.created_at,
            }
            for turn in snapshot.memory_turns
        ],
        "memory_raw_tail": [
            {
                "role": turn.role,
                "content": turn.content,
                "created_at": turn.created_at,
            }
            for turn in snapshot.memory_raw_tail
        ],
        "memory_ledger": [
            {
                "kind": item.kind,
                "content": item.content,
                "created_at": item.created_at,
                "source": item.source,
                "metadata": dict(item.metadata),
            }
            for item in snapshot.memory_ledger
        ],
        "memory_search_results": [
            {
                "question": item.question,
                "answer": item.answer,
                "sources": list(item.sources),
                "created_at": item.created_at,
                "location_hint": item.location_hint,
                "date_context": item.date_context,
            }
            for item in snapshot.memory_search_results
        ],
        "memory_state": {
            "active_topic": snapshot.memory_state.active_topic,
            "last_user_goal": snapshot.memory_state.last_user_goal,
            "pending_printable": snapshot.memory_state.pending_printable,
            "last_search_summary": snapshot.memory_state.last_search_summary,
            "open_loops": list(snapshot.memory_state.open_loops),
        },
    }


def _normalize_runtime_turns(
    turns: Iterable[object],
    *,
    field_name: str,
    max_items: int = _MAX_MEMORY_TURNS,
) -> tuple[RuntimeSnapshotTurn, ...]:
    return tuple(
        RuntimeSnapshotTurn(
            role=_trimmed_str(_item_value(turn, "role", ""), max_chars=_MAX_ROLE_CHARS) or "",
            content=_coerce_text(_item_value(turn, "content", ""), max_chars=_MAX_TURN_CONTENT_CHARS),
            created_at=_coerce_required_datetime_string(
                _item_value(turn, "created_at"),
                field_name=f"{field_name}.created_at",
            ),
        )
        for turn in _tail_limited_iterable(turns, max_items=max_items)
    )


def _normalize_runtime_ledger(
    items: Iterable[object],
    *,
    max_items: int = _MAX_MEMORY_LEDGER,
) -> tuple[RuntimeSnapshotLedgerItem, ...]:
    return tuple(
        RuntimeSnapshotLedgerItem(
            kind=_trimmed_str(_item_value(item, "kind", ""), max_chars=_MAX_KIND_CHARS) or "",
            content=_coerce_text(_item_value(item, "content", ""), max_chars=_MAX_LEDGER_CONTENT_CHARS),
            created_at=_coerce_required_datetime_string(
                _item_value(item, "created_at"),
                field_name="memory_ledger.created_at",
            ),
            source=_trimmed_str(_item_value(item, "source", "conversation"), max_chars=_MAX_SOURCE_LABEL_CHARS)
            or "conversation",
            metadata=_coerce_string_dict(
                _item_value(item, "metadata", {}),
                max_items=_MAX_METADATA_ITEMS,
                key_max_chars=_MAX_METADATA_KEY_CHARS,
                value_max_chars=_MAX_METADATA_VALUE_CHARS,
            ),
        )
        for item in _tail_limited_iterable(items, max_items=max_items)
    )


def _normalize_runtime_search_results(
    items: Iterable[object],
    *,
    max_items: int = _MAX_MEMORY_SEARCH_RESULTS,
) -> tuple[RuntimeSnapshotSearchEntry, ...]:
    return tuple(
        RuntimeSnapshotSearchEntry(
            question=_coerce_text(_item_value(item, "question", ""), max_chars=_MAX_SEARCH_QUESTION_CHARS),
            answer=_coerce_text(_item_value(item, "answer", ""), max_chars=_MAX_SEARCH_ANSWER_CHARS),
            sources=_coerce_string_tuple(
                _item_value(item, "sources", ()),
                max_items=_MAX_SOURCES_PER_SEARCH_ENTRY,
                item_max_chars=_MAX_SOURCE_TEXT_CHARS,
            ),
            created_at=_coerce_required_datetime_string(
                _item_value(item, "created_at"),
                field_name="memory_search_results.created_at",
            ),
            location_hint=_trimmed_str(
                _item_value(item, "location_hint"),
                max_chars=_MAX_SEARCH_CONTEXT_CHARS,
            ),
            date_context=_trimmed_str(
                _item_value(item, "date_context"),
                max_chars=_MAX_SEARCH_CONTEXT_CHARS,
            ),
        )
        for item in _tail_limited_iterable(items, max_items=max_items)
    )


def _normalize_memory_state(memory_state: MemoryState | Mapping[str, object] | None) -> RuntimeSnapshotMemoryState:
    if memory_state is None:
        return RuntimeSnapshotMemoryState()
    return RuntimeSnapshotMemoryState(
        active_topic=_trimmed_str(_item_value(memory_state, "active_topic"), max_chars=_MAX_TOPIC_CHARS),
        last_user_goal=_trimmed_str(_item_value(memory_state, "last_user_goal"), max_chars=_MAX_TOPIC_CHARS),
        pending_printable=_trimmed_str(
            _item_value(memory_state, "pending_printable"),
            max_chars=_MAX_TOPIC_CHARS,
        ),
        last_search_summary=_trimmed_str(
            _item_value(memory_state, "last_search_summary"),
            max_chars=_MAX_TOPIC_CHARS,
        ),
        open_loops=_coerce_string_tuple(
            _item_value(memory_state, "open_loops", ()),
            max_items=_MAX_OPEN_LOOPS,
            item_max_chars=_MAX_LOOP_CHARS,
        ),
    )


def _item_value(item: object, key: str, default: object = None) -> object:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _runtime_snapshot_from_mapping(
    data: Mapping[str, object],
    *,
    schema_version: int,
    revision: int,
) -> RuntimeSnapshot:
    status = _trimmed_str(data.get("status"), max_chars=_MAX_STATUS_CHARS) or _DEFAULT_STATUS
    return RuntimeSnapshot(
        status=status,
        printing_active=_normalize_printing_active(
            status=status,
            printing_active=data.get("printing_active", False),
        ),
        last_transcript=_coerce_optional_text(data.get("last_transcript"), max_chars=_MAX_TRANSCRIPT_CHARS),
        last_response=_coerce_optional_text(data.get("last_response"), max_chars=_MAX_RESPONSE_CHARS),
        updated_at=_coerce_optional_datetime_string(data.get("updated_at")),
        error_message=_coerce_optional_text(data.get("error_message"), max_chars=_MAX_ERROR_CHARS),
        user_voice_status=_trimmed_str(data.get("user_voice_status"), max_chars=_MAX_STATUS_CHARS),
        user_voice_confidence=_coerce_confidence(data.get("user_voice_confidence")),
        user_voice_checked_at=_coerce_optional_datetime_string(data.get("user_voice_checked_at")),
        user_voice_user_id=_trimmed_str(data.get("user_voice_user_id"), max_chars=_MAX_ID_CHARS),
        user_voice_user_display_name=_trimmed_str(
            data.get("user_voice_user_display_name"),
            max_chars=_MAX_NAME_CHARS,
        ),
        user_voice_match_source=_trimmed_str(
            data.get("user_voice_match_source"),
            max_chars=_MAX_SOURCE_LABEL_CHARS,
        ),
        voice_quiet_until_utc=_coerce_optional_datetime_string(data.get("voice_quiet_until_utc")),
        voice_quiet_reason=_coerce_optional_text(data.get("voice_quiet_reason"), max_chars=_MAX_REASON_CHARS),
        memory_turns=tuple(
            turn
            for turn in (
                _runtime_snapshot_turn(item)
                for item in _coerce_sequence(data.get("memory_turns"), max_items=_MAX_MEMORY_TURNS, tail=True)
            )
            if turn is not None
        ),
        memory_raw_tail=tuple(
            turn
            for turn in (
                _runtime_snapshot_turn(item)
                for item in _coerce_sequence(data.get("memory_raw_tail"), max_items=_MAX_MEMORY_RAW_TAIL, tail=True)
            )
            if turn is not None
        ),
        memory_ledger=tuple(
            item
            for item in (
                _runtime_snapshot_ledger_item(item)
                for item in _coerce_sequence(data.get("memory_ledger"), max_items=_MAX_MEMORY_LEDGER, tail=True)
            )
            if item is not None
        ),
        memory_search_results=tuple(
            item
            for item in (
                _runtime_snapshot_search_entry(item)
                for item in _coerce_sequence(
                    data.get("memory_search_results"),
                    max_items=_MAX_MEMORY_SEARCH_RESULTS,
                    tail=True,
                )
            )
            if item is not None
        ),
        memory_state=_runtime_snapshot_memory_state(data.get("memory_state")),
        schema_version=_coerce_non_negative_int(schema_version, default=_CURRENT_SCHEMA_VERSION),
        revision=_coerce_non_negative_int(revision, default=0),
    )


def _coerce_sequence(
    value: object,
    *,
    max_items: int | None = None,
    tail: bool = False,
) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    if max_items is None or len(value) <= max_items:
        return tuple(value)
    selected = value[-max_items:] if tail else value[:max_items]
    return tuple(selected)


def _tail_limited_iterable(value: object, *, max_items: int) -> tuple[object, ...]:
    if max_items <= 0 or value is None:
        return ()
    if isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        return ()
    if isinstance(value, Sequence):
        if len(value) <= max_items:
            return tuple(value)
        return tuple(value[-max_items:])
    try:
        return tuple(deque(cast(Iterable[object], value), maxlen=max_items))
    except TypeError:
        return ()


def _coerce_text(value: object, *, max_chars: int) -> str:
    text = "" if value is None else str(value)
    return _truncate_text(text, max_chars=max_chars)


def _coerce_optional_text(value: object, *, max_chars: int) -> str | None:
    if value is None:
        return None
    return _truncate_text(str(value), max_chars=max_chars)


def _trimmed_str(value: object, *, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _truncate_text(text, max_chars=max_chars)


def _truncate_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = " …[truncated]… "
    if max_chars <= len(marker):
        return text[:max_chars]
    keep_each_side = (max_chars - len(marker)) // 2
    remainder = max_chars - len(marker) - keep_each_side
    return f"{text[:keep_each_side]}{marker}{text[-remainder:]}"


def _coerce_string_dict(
    value: object,
    *,
    max_items: int,
    key_max_chars: int,
    value_max_chars: int,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}

    items = tuple(deque(value.items(), maxlen=max_items))
    normalized: dict[str, str] = {}
    for key, item_value in items:
        key_text = _trimmed_str(key, max_chars=key_max_chars)
        if key_text is None or item_value is None:
            continue
        value_text = _trimmed_str(item_value, max_chars=value_max_chars)
        if value_text is None:
            continue
        normalized[key_text] = value_text
    return normalized


def _coerce_string_tuple(
    value: object,
    *,
    max_items: int,
    item_max_chars: int,
) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in _tail_limited_iterable(value, max_items=max_items):
        if item is None:
            continue
        text = _trimmed_str(item, max_chars=item_max_chars)
        if text is not None:
            normalized.append(text)
    return tuple(normalized)


def _coerce_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(cast(_FloatLike, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


def _parse_datetime_like(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None

    return _normalize_datetime(parsed)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_optional_datetime_string(value: object) -> str | None:
    parsed = _parse_datetime_like(value)
    if parsed is None:
        return None
    return parsed.isoformat()


def _coerce_required_datetime_string(value: object, *, field_name: str) -> str:
    parsed = _parse_datetime_like(value)
    if parsed is None:
        raise RuntimeSnapshotStoreError(
            f"{field_name} must be a datetime or an ISO-8601 datetime string."
        )
    return parsed.isoformat()


def _coerce_non_negative_int(value: object, *, default: int) -> int:
    try:
        number = int(cast(_IntLike, value))
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _normalize_printing_active(*, status: object, printing_active: object) -> bool:
    normalized_status = _trimmed_str(status, max_chars=_MAX_STATUS_CHARS) or _DEFAULT_STATUS
    return bool(printing_active) or normalized_status == "printing"


def _runtime_snapshot_turn(item: object) -> RuntimeSnapshotTurn | None:
    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotTurn(
        role=_trimmed_str(item.get("role"), max_chars=_MAX_ROLE_CHARS) or "",
        content=_coerce_text(item.get("content"), max_chars=_MAX_TURN_CONTENT_CHARS),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
    )


def _runtime_snapshot_ledger_item(item: object) -> RuntimeSnapshotLedgerItem | None:
    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotLedgerItem(
        kind=_trimmed_str(item.get("kind"), max_chars=_MAX_KIND_CHARS) or "",
        content=_coerce_text(item.get("content"), max_chars=_MAX_LEDGER_CONTENT_CHARS),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
        source=_trimmed_str(item.get("source"), max_chars=_MAX_SOURCE_LABEL_CHARS) or "conversation",
        metadata=_coerce_string_dict(
            item.get("metadata"),
            max_items=_MAX_METADATA_ITEMS,
            key_max_chars=_MAX_METADATA_KEY_CHARS,
            value_max_chars=_MAX_METADATA_VALUE_CHARS,
        ),
    )


def _runtime_snapshot_search_entry(item: object) -> RuntimeSnapshotSearchEntry | None:
    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotSearchEntry(
        question=_coerce_text(item.get("question"), max_chars=_MAX_SEARCH_QUESTION_CHARS),
        answer=_coerce_text(item.get("answer"), max_chars=_MAX_SEARCH_ANSWER_CHARS),
        sources=_coerce_string_tuple(
            item.get("sources"),
            max_items=_MAX_SOURCES_PER_SEARCH_ENTRY,
            item_max_chars=_MAX_SOURCE_TEXT_CHARS,
        ),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
        location_hint=_trimmed_str(item.get("location_hint"), max_chars=_MAX_SEARCH_CONTEXT_CHARS),
        date_context=_trimmed_str(item.get("date_context"), max_chars=_MAX_SEARCH_CONTEXT_CHARS),
    )


def _runtime_snapshot_memory_state(value: object) -> RuntimeSnapshotMemoryState:
    if not isinstance(value, Mapping):
        return RuntimeSnapshotMemoryState()
    return RuntimeSnapshotMemoryState(
        active_topic=_optional_str(value, "active_topic", max_chars=_MAX_TOPIC_CHARS),
        last_user_goal=_optional_str(value, "last_user_goal", max_chars=_MAX_TOPIC_CHARS),
        pending_printable=_optional_str(value, "pending_printable", max_chars=_MAX_TOPIC_CHARS),
        last_search_summary=_optional_str(value, "last_search_summary", max_chars=_MAX_TOPIC_CHARS),
        open_loops=_coerce_string_tuple(
            value.get("open_loops"),
            max_items=_MAX_OPEN_LOOPS,
            item_max_chars=_MAX_LOOP_CHARS,
        ),
    )


def _optional_str(data: Mapping[str, object] | None, key: str, *, max_chars: int) -> str | None:
    if not isinstance(data, Mapping):
        return None
    return _trimmed_str(data.get(key), max_chars=max_chars)
