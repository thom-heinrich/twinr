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


# AUDIT-FIX(#11): Collapse heterogeneous storage failures into one stable exception type.
class RuntimeSnapshotStoreError(RuntimeError):
    """Raised when the runtime snapshot store cannot safely read or write state."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotTurn:
    role: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotLedgerItem:
    kind: str
    content: str
    created_at: str
    source: str
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotSearchEntry:
    question: str
    answer: str
    sources: tuple[str, ...]
    created_at: str
    location_hint: str | None = None
    date_context: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSnapshotMemoryState:
    active_topic: str | None = None
    last_user_goal: str | None = None
    pending_printable: str | None = None
    last_search_summary: str | None = None
    open_loops: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    status: str = _DEFAULT_STATUS
    last_transcript: str | None = None
    last_response: str | None = None
    updated_at: str | None = None
    error_message: str | None = None
    user_voice_status: str | None = None
    user_voice_confidence: float | None = None
    user_voice_checked_at: str | None = None
    memory_turns: tuple[RuntimeSnapshotTurn, ...] = ()
    memory_raw_tail: tuple[RuntimeSnapshotTurn, ...] = ()
    memory_ledger: tuple[RuntimeSnapshotLedgerItem, ...] = ()
    memory_search_results: tuple[RuntimeSnapshotSearchEntry, ...] = ()
    # AUDIT-FIX(#10): Avoid a shared class-level default instance for nested state.
    memory_state: RuntimeSnapshotMemoryState = field(default_factory=RuntimeSnapshotMemoryState)

    @property
    def memory_count(self) -> int:
        return len(self.memory_turns)


class RuntimeSnapshotStore:
    def __init__(self, path: str | Path) -> None:
        # AUDIT-FIX(#7): Normalize the configured path early and reject directory-like targets.
        self.path = Path(path).expanduser()
        if not self.path.name or self.path.name in {".", ".."}:
            raise RuntimeSnapshotStoreError("Runtime snapshot path must point to a file.")
        self._backup_path = self.path.with_name(f"{self.path.name}{_BACKUP_SUFFIX}")
        self._lock_path = self.path.with_name(f".{self.path.name}{_LOCK_SUFFIX}")

    def load(self) -> RuntimeSnapshot:
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
    ) -> RuntimeSnapshot:
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
                os.fsync(handle.fileno())
                temp_file_path = Path(handle.name)

            os.replace(temp_file_path, target)
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
        try:
            self._atomic_write_text(self._backup_path, payload_text)
        except RuntimeSnapshotStoreError:
            # AUDIT-FIX(#5): The primary snapshot is already durable; keep backup refresh best-effort.
            return

    def _fsync_directory(self, directory: Path) -> None:
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
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


# AUDIT-FIX(#8): Normalize free-text fields to the declared string API.
def _coerce_text(value: object) -> str:
    return "" if value is None else str(value)


def _coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _trimmed_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_string_dict(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item_value)
        for key, item_value in value.items()
        if str(item_value or "").strip()
    }


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    return tuple(
        str(item)
        for item in _coerce_sequence(value)
        if str(item or "").strip()
    )


# AUDIT-FIX(#3): Accept only finite 0..1 confidence values.
def _coerce_confidence(value: object) -> float | None:
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


def _datetime_to_utc_iso(value: object, *, field_name: str) -> str:
    if not isinstance(value, datetime):
        raise RuntimeSnapshotStoreError(f"{field_name} must be a datetime.")
    return _normalize_datetime(value).isoformat()


def _runtime_snapshot_turn(item: object) -> RuntimeSnapshotTurn | None:
    if not isinstance(item, Mapping):
        return None
    return RuntimeSnapshotTurn(
        role=_coerce_text(item.get("role")),
        content=_coerce_text(item.get("content")),
        created_at=_coerce_optional_datetime_string(item.get("created_at")) or "",
    )


def _runtime_snapshot_ledger_item(item: object) -> RuntimeSnapshotLedgerItem | None:
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
    if not isinstance(data, Mapping):
        return None
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None