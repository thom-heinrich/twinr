"""Persist and restore normalized runtime snapshots for the base Twinr agent.

This module defines the snapshot schema shared by runtime, display, web, and
ops code and provides a file-backed store that reads malformed data
defensively. Import the dataclasses when consuming persisted runtime state and
use ``RuntimeSnapshotStore`` for all snapshot I/O.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import math
import os
import tempfile

from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry


_DEFAULT_STATUS = "waiting"
_BACKUP_SUFFIX = ".bak"
_LOCK_SUFFIX = ".lock"
_CROSS_SERVICE_READ_MODE = 0o644


# AUDIT-FIX(#11): Collapse heterogeneous storage failures into one stable exception type.
class RuntimeSnapshotStoreError(RuntimeError):
    """Raised when the runtime snapshot store cannot safely read or write state."""


def _utcnow() -> datetime:
    """Return the current UTC time as an aware ``datetime``."""

    return datetime.now(timezone.utc)


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
    """Represent the normalized runtime snapshot payload.

    Attributes:
        status: Canonical runtime status string.
        last_transcript: Most recent transcript text, if any.
        last_response: Most recent assistant response, if any.
        updated_at: Snapshot write timestamp in UTC ISO-8601 format.
        error_message: Last normalized runtime error, if any.
        user_voice_status: Last user-voice classification label, if any.
        user_voice_confidence: Last normalized confidence score in ``0..1``.
        user_voice_checked_at: Timestamp for the voice-status check.
        user_voice_user_id: Matched enrolled local household member id, if any.
        user_voice_user_display_name: Human-facing name for the matched member, if any.
        user_voice_match_source: Internal source tag for the current match, if any.
        voice_quiet_until_utc: Optional UTC deadline until transcript-first
            voice wake should stay quiet.
        voice_quiet_reason: Optional short operator/user-facing reason for the
            temporary quiet window.
        memory_turns: Canonical conversation turns kept in active memory.
        memory_raw_tail: Unsummarized recent turns kept alongside memory.
        memory_ledger: Structured ledger entries derived from memory events.
        memory_search_results: Structured verified-search memory entries.
        memory_state: Structured memory-sidecar state.
    """

    status: str = _DEFAULT_STATUS
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
    # AUDIT-FIX(#10): Avoid a shared class-level default instance for nested state.
    memory_state: RuntimeSnapshotMemoryState = field(default_factory=RuntimeSnapshotMemoryState)

    @property
    def memory_count(self) -> int:
        """Return the number of canonical active-memory turns."""

        return len(self.memory_turns)


class RuntimeSnapshotStore:
    """Load and save runtime snapshots with locking and atomic writes.

    The store uses a sidecar lock file and a backup copy so concurrent readers
    and writers can recover cleanly from partial writes or corrupted primary
    snapshot files on the Pi.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the store for one runtime snapshot path.

        Args:
            path: Primary snapshot file path.

        Raises:
            RuntimeSnapshotStoreError: If the configured path does not point to
                a file target.
        """

        # AUDIT-FIX(#7): Normalize the configured path early and reject directory-like targets.
        self.path = Path(path).expanduser()
        if not self.path.name or self.path.name in {".", ".."}:
            raise RuntimeSnapshotStoreError("Runtime snapshot path must point to a file.")
        self._backup_path = self.path.with_name(f"{self.path.name}{_BACKUP_SUFFIX}")
        self._lock_path = self.path.with_name(f".{self.path.name}{_LOCK_SUFFIX}")

    def load(self) -> RuntimeSnapshot:
        """Load the most recent valid snapshot from primary or backup storage.

        Returns:
            The normalized snapshot payload. If no valid snapshot exists, an
            empty default ``RuntimeSnapshot`` is returned.

        Raises:
            RuntimeSnapshotStoreError: If filesystem validation or locking
                fails while attempting the load.
        """

        if not self.path.parent.exists():
            return RuntimeSnapshot()

        # AUDIT-FIX(#6): Serialize readers and writers with a sidecar advisory lock.
        with self._locked(shared=True):
            # AUDIT-FIX(#7): Refuse unsafe targets before touching the filesystem.
            self._validate_targets()

            candidates = tuple(path for path in (self.path, self._backup_path) if path.exists())
            if not candidates:
                return RuntimeSnapshot()

            for candidate in candidates:
                try:
                    snapshot = self._load_from_path(candidate)
                except RuntimeSnapshotStoreError:
                    raise
                if snapshot is not None:
                    return snapshot

        return RuntimeSnapshot()

    def save(
        self,
        *,
        status: str,
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
    ) -> RuntimeSnapshot:
        """Normalize and persist one runtime snapshot payload.

        Args:
            status: Canonical runtime status string.
            memory_turns: Active conversation turns to persist.
            memory_raw_tail: Optional unsummarized recent turns.
            memory_ledger: Optional structured ledger entries.
            memory_search_results: Optional structured search-memory entries.
            memory_state: Optional structured memory-sidecar state.
            last_transcript: Most recent user transcript, if any.
            last_response: Most recent assistant response, if any.
            error_message: Most recent runtime error, if any.
            user_voice_status: Last voice-status label, if any.
            user_voice_confidence: Last voice-status confidence, if any.
            user_voice_checked_at: Timestamp for the voice-status observation.
            user_voice_user_id: Matched enrolled local household member id, if any.
            user_voice_user_display_name: Human-facing name for the matched member, if any.
            user_voice_match_source: Internal source tag for the current match, if any.
            voice_quiet_until_utc: Optional UTC deadline for the temporary
                voice-quiet window.
            voice_quiet_reason: Optional short reason for that quiet window.

        Returns:
            The normalized snapshot object that was written to disk.

        Raises:
            RuntimeSnapshotStoreError: If the payload cannot be serialized or
                durably written.
        """

        # AUDIT-FIX(#3): Clamp/sanitize confidence values before they can crash load or emit invalid JSON.
        sanitized_confidence = _coerce_confidence(user_voice_confidence)

        snapshot = RuntimeSnapshot(
            status=_trimmed_str(status) or _DEFAULT_STATUS,
            last_transcript=_coerce_optional_text(last_transcript),
            last_response=_coerce_optional_text(last_response),
            updated_at=_utcnow().isoformat(),
            error_message=_coerce_optional_text(error_message),
            user_voice_status=_trimmed_str(user_voice_status),
            user_voice_confidence=sanitized_confidence,
            # AUDIT-FIX(#4): Normalize datetime-like timestamps to UTC for DST-safe persistence.
            user_voice_checked_at=_coerce_optional_datetime_string(user_voice_checked_at),
            user_voice_user_id=_trimmed_str(user_voice_user_id),
            user_voice_user_display_name=_trimmed_str(user_voice_user_display_name),
            user_voice_match_source=_trimmed_str(user_voice_match_source),
            voice_quiet_until_utc=_coerce_optional_datetime_string(voice_quiet_until_utc),
            voice_quiet_reason=_coerce_optional_text(voice_quiet_reason),
            memory_turns=tuple(
                RuntimeSnapshotTurn(
                    role=_coerce_text(getattr(turn, "role", "")),
                    content=_coerce_text(getattr(turn, "content", "")),
                    created_at=_datetime_to_utc_iso(
                        getattr(turn, "created_at", None),
                        field_name="memory_turns.created_at",
                    ),
                )
                for turn in tuple(memory_turns)
            ),
            memory_raw_tail=tuple(
                RuntimeSnapshotTurn(
                    role=_coerce_text(getattr(turn, "role", "")),
                    content=_coerce_text(getattr(turn, "content", "")),
                    created_at=_datetime_to_utc_iso(
                        getattr(turn, "created_at", None),
                        field_name="memory_raw_tail.created_at",
                    ),
                )
                for turn in tuple(memory_raw_tail or ())
            ),
            memory_ledger=tuple(
                RuntimeSnapshotLedgerItem(
                    kind=_coerce_text(getattr(item, "kind", "")),
                    content=_coerce_text(getattr(item, "content", "")),
                    created_at=_datetime_to_utc_iso(
                        getattr(item, "created_at", None),
                        field_name="memory_ledger.created_at",
                    ),
                    source=_trimmed_str(getattr(item, "source", "conversation")) or "conversation",
                    metadata=_coerce_string_dict(getattr(item, "metadata", {})),
                )
                for item in tuple(memory_ledger or ())
            ),
            memory_search_results=tuple(
                RuntimeSnapshotSearchEntry(
                    question=_coerce_text(getattr(item, "question", "")),
                    answer=_coerce_text(getattr(item, "answer", "")),
                    sources=_coerce_string_tuple(getattr(item, "sources", ())),
                    created_at=_datetime_to_utc_iso(
                        getattr(item, "created_at", None),
                        field_name="memory_search_results.created_at",
                    ),
                    location_hint=_trimmed_str(getattr(item, "location_hint", None)),
                    date_context=_trimmed_str(getattr(item, "date_context", None)),
                )
                for item in tuple(memory_search_results or ())
            ),
            memory_state=RuntimeSnapshotMemoryState(
                active_topic=_trimmed_str(getattr(memory_state, "active_topic", None))
                if memory_state is not None
                else None,
                last_user_goal=_trimmed_str(getattr(memory_state, "last_user_goal", None))
                if memory_state is not None
                else None,
                pending_printable=_trimmed_str(getattr(memory_state, "pending_printable", None))
                if memory_state is not None
                else None,
                last_search_summary=_trimmed_str(getattr(memory_state, "last_search_summary", None))
                if memory_state is not None
                else None,
                open_loops=_coerce_string_tuple(getattr(memory_state, "open_loops", ()))
                if memory_state is not None
                else (),
            ),
        )

        payload = {
            "status": snapshot.status,
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

        # AUDIT-FIX(#11): Surface write failures through one stable exception type instead of raw OSError/ValueError.
        try:
            # AUDIT-FIX(#3): Enforce strict JSON output so NaN/Infinity never reach disk.
            payload_text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeSnapshotStoreError("Unable to serialize the runtime snapshot.") from exc

        # AUDIT-FIX(#7): Validate the path chain before directory creation to avoid unsafe targets.
        self._ensure_parent_directory()

        # AUDIT-FIX(#6): Serialize writers so concurrent saves cannot interleave or overwrite each other mid-flight.
        with self._locked(shared=False):
            self._validate_targets()
            # AUDIT-FIX(#2): Use durable atomic writes so power loss cannot silently drop the most recent snapshot.
            self._atomic_write_text(self.path, payload_text)
            # AUDIT-FIX(#5): Keep a validated sidecar backup for recovery from primary corruption.
            self._write_backup_best_effort(payload_text)

        return snapshot

    def _load_from_path(self, candidate: Path) -> RuntimeSnapshot | None:
        """Load and normalize one snapshot file candidate if it is valid."""

        try:
            payload_text = candidate.read_text(encoding="utf-8")
            data = json.loads(payload_text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

        # AUDIT-FIX(#1): Reject malformed JSON shapes instead of crashing on .get()/iteration below.
        if not isinstance(data, Mapping):
            return None

        return RuntimeSnapshot(
            status=_trimmed_str(data.get("status")) or _DEFAULT_STATUS,
            # AUDIT-FIX(#8): Coerce optional text fields to the declared API type.
            last_transcript=_coerce_optional_text(data.get("last_transcript")),
            last_response=_coerce_optional_text(data.get("last_response")),
            updated_at=_coerce_optional_datetime_string(data.get("updated_at")),
            error_message=_coerce_optional_text(data.get("error_message")),
            user_voice_status=_trimmed_str(data.get("user_voice_status")),
            user_voice_confidence=_coerce_confidence(data.get("user_voice_confidence")),
            user_voice_checked_at=_coerce_optional_datetime_string(data.get("user_voice_checked_at")),
            user_voice_user_id=_trimmed_str(data.get("user_voice_user_id")),
            user_voice_user_display_name=_trimmed_str(data.get("user_voice_user_display_name")),
            user_voice_match_source=_trimmed_str(data.get("user_voice_match_source")),
            voice_quiet_until_utc=_coerce_optional_datetime_string(data.get("voice_quiet_until_utc")),
            voice_quiet_reason=_coerce_optional_text(data.get("voice_quiet_reason")),
            memory_turns=tuple(
                turn
                for turn in (_runtime_snapshot_turn(item) for item in _coerce_sequence(data.get("memory_turns")))
                if turn is not None
            ),
            memory_raw_tail=tuple(
                turn
                for turn in (_runtime_snapshot_turn(item) for item in _coerce_sequence(data.get("memory_raw_tail")))
                if turn is not None
            ),
            memory_ledger=tuple(
                item
                for item in (
                    _runtime_snapshot_ledger_item(item)
                    for item in _coerce_sequence(data.get("memory_ledger"))
                )
                if item is not None
            ),
            memory_search_results=tuple(
                item
                for item in (
                    _runtime_snapshot_search_entry(item)
                    for item in _coerce_sequence(data.get("memory_search_results"))
                )
                if item is not None
            ),
            memory_state=_runtime_snapshot_memory_state(data.get("memory_state")),
        )

    def _ensure_parent_directory(self) -> None:
        """Create the snapshot parent directory after validating its path."""

        self._validate_parent_chain(self.path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeSnapshotStoreError("Unable to create the runtime snapshot directory.") from exc

    def _validate_targets(self) -> None:
        """Validate the primary, backup, and lock-file targets."""

        for target in (self.path, self._backup_path, self._lock_path):
            self._validate_parent_chain(target)
            self._validate_target_path(target)

    def _validate_target_path(self, target: Path) -> None:
        """Reject symlinked or non-file snapshot targets."""

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
        """Reject symlinked or non-directory parents for a target path."""

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
        """Hold the advisory snapshot sidecar lock for one operation."""

        lock_handle = None
        try:
            # AUDIT-FIX(#7): Validate the lock sidecar before opening it so symlinked lock files cannot be followed.
            self._validate_parent_chain(self._lock_path)
            self._validate_target_path(self._lock_path)
            lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            lock_fd = os.open(self._lock_path, lock_flags, 0o600)
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

    def _atomic_write_text(self, target: Path, payload_text: str) -> None:
        """Atomically replace ``target`` with serialized snapshot text."""

        temp_file_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload_text)
                handle.flush()
                os.fchmod(handle.fileno(), _CROSS_SERVICE_READ_MODE)
                os.fsync(handle.fileno())
                temp_file_path = Path(handle.name)

            os.replace(temp_file_path, target)
            os.chmod(target, _CROSS_SERVICE_READ_MODE)
            # AUDIT-FIX(#9): Clear the temp path after replace so cleanup cannot unlink a reused filename.
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

    def _write_backup_best_effort(self, payload_text: str) -> None:
        """Refresh the backup snapshot file without failing the main save."""

        try:
            self._atomic_write_text(self._backup_path, payload_text)
        except RuntimeSnapshotStoreError:
            # AUDIT-FIX(#5): The primary snapshot is already durable; keep backup refresh best-effort.
            return

    def _fsync_directory(self, directory: Path) -> None:
        """Fsync the directory that owns the snapshot files."""

        try:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
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


# AUDIT-FIX(#1): Treat malformed JSON collection fields as empty sequences instead of crashing.
def _coerce_sequence(value: object) -> tuple[object, ...]:
    """Coerce a JSON-like collection field to an immutable sequence."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


# AUDIT-FIX(#8): Normalize free-text fields to the declared string API.
def _coerce_text(value: object) -> str:
    """Coerce a free-text field to the declared non-optional string type."""

    return "" if value is None else str(value)


def _coerce_optional_text(value: object) -> str | None:
    """Coerce a free-text field to ``str | None``."""

    if value is None:
        return None
    return str(value)


def _trimmed_str(value: object) -> str | None:
    """Return a stripped string or ``None`` for blank input."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_string_dict(value: object) -> dict[str, str]:
    """Coerce mapping-like metadata to ``dict[str, str]``."""

    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item_value)
        for key, item_value in value.items()
        if str(item_value or "").strip()
    }


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    """Coerce a sequence-like field to a tuple of non-blank strings."""

    return tuple(
        str(item)
        for item in _coerce_sequence(value)
        if str(item or "").strip()
    )


# AUDIT-FIX(#3): Accept only finite 0..1 confidence values.
def _coerce_confidence(value: object) -> float | None:
    """Coerce a confidence field to a finite value in ``0..1``."""

    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    if confidence < 0.0 or confidence > 1.0:
        return None
    return confidence


# AUDIT-FIX(#4): Canonicalize datetime-like values to UTC ISO-8601 on read and write.
def _parse_datetime_like(value: object) -> datetime | None:
    """Parse supported datetime-like inputs and normalize them to UTC."""

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
    """Normalize a ``datetime`` to an aware UTC value."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _coerce_optional_datetime_string(value: object) -> str | None:
    """Coerce a datetime-like value to a UTC ISO-8601 string."""

    parsed = _parse_datetime_like(value)
    if parsed is None:
        return None
    return parsed.isoformat()


def _datetime_to_utc_iso(value: object, *, field_name: str) -> str:
    """Convert a required ``datetime`` field to a UTC ISO-8601 string."""

    if not isinstance(value, datetime):
        raise RuntimeSnapshotStoreError(f"{field_name} must be a datetime.")
    return _normalize_datetime(value).isoformat()


def _runtime_snapshot_turn(item: object) -> RuntimeSnapshotTurn | None:
    """Coerce one mapping-like payload item to ``RuntimeSnapshotTurn``."""

    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotTurn(
        role=_coerce_text(item.get("role")),
        content=_coerce_text(item.get("content")),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
    )


def _runtime_snapshot_ledger_item(item: object) -> RuntimeSnapshotLedgerItem | None:
    """Coerce one mapping-like payload item to ``RuntimeSnapshotLedgerItem``."""

    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotLedgerItem(
        kind=_coerce_text(item.get("kind")),
        content=_coerce_text(item.get("content")),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
        source=_trimmed_str(item.get("source")) or "conversation",
        metadata=_coerce_string_dict(item.get("metadata")),
    )


def _runtime_snapshot_search_entry(item: object) -> RuntimeSnapshotSearchEntry | None:
    """Coerce one mapping-like payload item to ``RuntimeSnapshotSearchEntry``."""

    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotSearchEntry(
        question=_coerce_text(item.get("question")),
        answer=_coerce_text(item.get("answer")),
        sources=_coerce_string_tuple(item.get("sources")),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
        location_hint=_trimmed_str(item.get("location_hint")),
        date_context=_trimmed_str(item.get("date_context")),
    )


def _runtime_snapshot_memory_state(value: object) -> RuntimeSnapshotMemoryState:
    """Coerce one mapping-like payload item to ``RuntimeSnapshotMemoryState``."""

    if not isinstance(value, Mapping):
        return RuntimeSnapshotMemoryState()
    return RuntimeSnapshotMemoryState(
        active_topic=_optional_str(value, "active_topic"),
        last_user_goal=_optional_str(value, "last_user_goal"),
        pending_printable=_optional_str(value, "pending_printable"),
        last_search_summary=_optional_str(value, "last_search_summary"),
        open_loops=_coerce_string_tuple(value.get("open_loops")),
    )


def _optional_str(data: Mapping[str, object] | None, key: str) -> str | None:
    """Return a stripped optional string field from a mapping."""

    if not isinstance(data, Mapping):
        return None
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None
