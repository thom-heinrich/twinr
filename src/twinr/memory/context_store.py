"""Persist Twinr prompt context and durable explicit memories.

This module owns the markdown-backed prompt-memory and managed-context stores
used by prompt assembly, the web dashboard, and long-term memory runtime
services. The stores support local file persistence plus remote-primary
snapshot migration when Twinr's remote memory backend is enabled.
"""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import TYPE_CHECKING, Iterator, Mapping  # AUDIT-FIX(#9): Keep runtime type-hint evaluation stable on Python 3.11.

from twinr.agent.base_agent.config import TwinrConfig
from twinr.text_utils import collapse_whitespace, slugify_identifier

if TYPE_CHECKING:
    from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

LOGGER = logging.getLogger(__name__)

_MANAGED_START = "<!-- TWINR_MANAGED_CONTEXT_START -->"
_MANAGED_END = "<!-- TWINR_MANAGED_CONTEXT_END -->"
_PROMPT_MEMORY_SCHEMA = "twinr_prompt_memory"
_PROMPT_MEMORY_VERSION = 1
_MANAGED_CONTEXT_SCHEMA = "twinr_managed_context"
_MANAGED_CONTEXT_VERSION = 1
_DEFAULT_MAX_ENTRIES = 24
_DEFAULT_RENDER_LIMIT = 12

_LOCK_REGISTRY: dict[str, threading.RLock] = {}
_LOCK_REGISTRY_GUARD = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_text(value: str, *, limit: int) -> str:
    text = collapse_whitespace(value)
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _slugify(value: str, *, fallback: str) -> str:
    return slugify_identifier(value, fallback=fallback)


def _parse_markdown_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("### "):
        return None
    entry_id = stripped[4:].strip()
    if not entry_id:
        return None
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(char not in allowed for char in entry_id):
        return None
    return entry_id


def _parse_markdown_field(line: str, *, allow_uppercase_key: bool = False) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    key, separator, value = stripped[2:].partition(":")
    if separator != ":":
        return None
    normalized_key = key.strip()
    if not normalized_key:
        return None
    valid_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    lowered = normalized_key.lower()
    if allow_uppercase_key:
        if any(char not in valid_chars for char in lowered):
            return None
        return lowered, value.strip()
    if any(char not in valid_chars for char in normalized_key):
        return None
    return normalized_key, value.strip()


def _coerce_utc(value: datetime) -> datetime:
    # AUDIT-FIX(#6): Normalize all timestamps to aware UTC to remove naive/aware drift and DST ambiguity.
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        return _utcnow()
    try:
        return _coerce_utc(datetime.fromisoformat(text))
    except ValueError:
        # AUDIT-FIX(#6): Keep parse failures deterministic and non-fatal instead of propagating mixed/invalid datetimes.
        LOGGER.warning("Falling back to current UTC time for invalid timestamp %r.", text)
        return _utcnow()


def _normalize_entry_id(value: str) -> str:
    stripped = str(value or "").strip().upper()
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if stripped and all(char in allowed for char in stripped):
        return stripped
    return f"MEM-{_utcnow().strftime('%Y%m%dT%H%M%S%fZ')}"


def _coerce_limit(value: int, *, default: int) -> int:
    # AUDIT-FIX(#10): Clamp externally supplied limits so zero/negative values cannot silently drop memory output.
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _get_named_lock(name: str) -> threading.RLock:
    with _LOCK_REGISTRY_GUARD:
        lock = _LOCK_REGISTRY.get(name)
        if lock is None:
            lock = threading.RLock()
            _LOCK_REGISTRY[name] = lock
        return lock


@contextmanager
def _acquire_named_locks(*names: str) -> Iterator[None]:
    # AUDIT-FIX(#5): Serialize read-modify-write flows across identical paths and remote snapshot kinds.
    unique_names = sorted(set(name for name in names if name))
    locks = [_get_named_lock(name) for name in unique_names]
    for lock in locks:
        lock.acquire()
    try:
        yield
    finally:
        for lock in reversed(locks):
            lock.release()


def _resolve_storage_path(
    path: str | Path,
    *,
    root_dir: str | Path | None = None,
    allow_absolute_outside_root: bool = True,
) -> Path:
    raw_path = Path(path).expanduser()
    if root_dir is None:
        return raw_path.resolve(strict=False)
    if raw_path.is_absolute() and allow_absolute_outside_root:
        return raw_path.resolve(strict=False)

    root_path = Path(root_dir).expanduser().resolve(strict=False)
    candidate = raw_path if raw_path.is_absolute() else root_path / raw_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        # AUDIT-FIX(#1): Keep config-derived storage paths inside the configured project root.
        raise ValueError(
            f"Storage path {str(candidate)!r} escapes the configured project root {str(root_path)!r}."
        ) from exc
    return resolved


def _fsync_directory(directory: Path) -> None:
    try:
        directory_fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


def _read_text_file(path: Path) -> str | None:
    try:
        # AUDIT-FIX(#1): Refuse surprising filesystem targets and degrade gracefully instead of crashing prompt assembly.
        if path.is_symlink():
            LOGGER.warning("Refusing to read symlinked Twinr storage file: %s", path)
            return None
        if not path.exists():
            return None
        if path.is_dir():
            LOGGER.warning("Refusing to read directory as Twinr storage file: %s", path)
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        LOGGER.warning("Failed to read Twinr storage file %s: %s", path, exc)
        return None


def _atomic_write_text(path: Path, text: str) -> None:
    # AUDIT-FIX(#1): Use temp-file + fsync + replace so writes survive power loss and do not tear in place.
    cross_service_read_mode = 0o644
    parent = path.parent.resolve(strict=False)
    parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Twinr storage path points to a directory: {path}")

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fchmod(handle.fileno(), cross_service_read_mode)
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
        os.chmod(path, cross_service_read_mode)
        _fsync_directory(parent)
    except Exception:
        with suppress(OSError):
            temp_path.unlink()
        raise


def _neutralize_reserved_marker_lines(value: str) -> str:
    adjusted_lines: list[str] = []
    for raw_line in value.split("\n"):
        stripped = raw_line.strip()
        if stripped in {_MANAGED_START, _MANAGED_END}:
            # AUDIT-FIX(#5): Stop exact managed-marker lines in base content from being reinterpreted as structural delimiters.
            adjusted_lines.append(f"`{stripped}`")
        else:
            adjusted_lines.append(raw_line)
    return "\n".join(adjusted_lines)


def _is_remote_unavailable_error(exc: Exception) -> bool:
    return type(exc).__name__ == "LongTermRemoteUnavailableError"


@dataclass(frozen=True, slots=True)
class ManagedContextEntry:
    """Store one Twinr-managed user or personality context update.

    Attributes:
        key: Stable normalized category identifier.
        instruction: Short bounded instruction injected into prompt context.
        updated_at: UTC timestamp used for ordering and snapshot persistence.
    """

    key: str
    instruction: str
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True, slots=True)
class PersistentMemoryEntry:
    """Store one explicit durable memory item saved for later turns.

    Attributes:
        entry_id: Stable identifier for persistence and deduplication.
        kind: Normalized memory category label.
        summary: Short bounded summary text.
        details: Optional longer detail text.
        created_at: UTC timestamp when the memory was first stored.
        updated_at: UTC timestamp of the latest update.
    """

    entry_id: str
    kind: str
    summary: str
    details: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


class ManagedContextFileStore:
    """Manage a markdown-backed prompt-context section with optional remote sync.

    The store keeps the human-authored base markdown text separate from the
    Twinr-managed updates section so operators can edit the base content while
    Twinr appends or migrates structured updates safely.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        section_title: str,
        remote_state: "LongTermRemoteStateStore | None" = None,
        remote_snapshot_kind: str | None = None,
        root_dir: str | Path | None = None,
    ) -> None:
        # AUDIT-FIX(#1): Resolve and constrain storage paths eagerly so later file ops hit a canonical location.
        self.path = _resolve_storage_path(path, root_dir=root_dir)
        self.section_title = section_title
        self.remote_state = remote_state
        self.remote_snapshot_kind = _normalize_text(remote_snapshot_kind or "", limit=80) or None
        # AUDIT-FIX(#5): Share locks across instances that target the same file or snapshot kind.
        self._lock_names = [f"path::{self.path}"]
        if self.remote_snapshot_kind:
            self._lock_names.append(f"snapshot::{self.remote_snapshot_kind}")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _acquire_named_locks(*self._lock_names):
            yield

    def _remote_enabled(self) -> bool:
        return bool(self.remote_state is not None and self.remote_state.enabled and self.remote_snapshot_kind)

    def _migration_enabled(self) -> bool:
        config = getattr(self.remote_state, "config", None)
        return bool(config is not None and getattr(config, "long_term_memory_migration_enabled", False))

    def load_base_text(self) -> str:
        """Return the human-authored markdown outside the managed block."""

        with self._locked():
            prefix, _managed_entries, _suffix = self._split_document()
            return prefix.strip()

    def load_entries(self) -> tuple[ManagedContextEntry, ...]:
        """Load managed context entries from remote or local storage.

        Returns:
            A tuple of normalized ``ManagedContextEntry`` objects.
        """

        with self._locked():
            if self._remote_enabled():
                # AUDIT-FIX(#2): Treat valid-empty, missing, invalid, and unavailable remote snapshots as different states.
                status, remote_entries = self._try_load_remote_entries()
                if status == "ok":
                    return remote_entries

                local_entries = self._load_local_entries()
                if local_entries and status in {"missing", "invalid"} and self._migration_enabled():
                    self._try_save_remote_entries(local_entries)
                return local_entries
            return self._load_local_entries()

    def upsert(self, *, category: str, instruction: str) -> ManagedContextEntry:
        """Create or replace one managed context instruction.

        Args:
            category: Stable category name such as ``response_style``.
            instruction: Short bounded instruction text to persist.

        Returns:
            The stored ``ManagedContextEntry``.

        Raises:
            ValueError: If the instruction is empty after normalization.
            RuntimeError: If the updated entry cannot be persisted.
        """

        key = _slugify(category, fallback="update")
        clean_instruction = _normalize_text(instruction, limit=220)
        if not clean_instruction:
            raise ValueError("instruction must not be empty")

        with self._locked():
            entries = list(self.load_entries())
            updated = ManagedContextEntry(key=key, instruction=clean_instruction, updated_at=_utcnow())
            for index, existing in enumerate(entries):
                if existing.key != key:
                    continue
                entries[index] = updated
                if not self._persist_entries(tuple(entries)):
                    raise RuntimeError(f"Failed to persist managed context update for {self.path}.")
                return updated
            entries.append(updated)
            if not self._persist_entries(tuple(entries)):
                raise RuntimeError(f"Failed to persist managed context update for {self.path}.")
            return updated

    def delete(self, *, category: str) -> ManagedContextEntry | None:
        """Remove one managed context instruction when it exists."""

        key = _slugify(category, fallback="update")
        with self._locked():
            entries = list(self.load_entries())
            for index, existing in enumerate(entries):
                if existing.key != key:
                    continue
                removed = entries.pop(index)
                if not self._persist_entries(tuple(entries)):
                    raise RuntimeError(f"Failed to persist managed context deletion for {self.path}.")
                return removed
        return None

    def replace_base_text(self, content: str) -> None:
        """Replace the human-authored markdown outside the managed block."""

        with self._locked():
            _prefix, _managed_entries, suffix = self._split_document()
            normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip()
            # AUDIT-FIX(#5): Prevent accidental marker collisions in base text while preserving readable markdown.
            safe_prefix = _neutralize_reserved_marker_lines(normalized)
            # Remote-primary keeps managed entries off-disk; only the base text stays in the local file.
            managed_entries = () if self._remote_enabled() else self.load_entries()
            self._write_document(prefix=safe_prefix, entries=managed_entries, suffix=suffix)

    def render_context(self) -> str | None:
        """Render base markdown and managed entries for prompt assembly."""

        with self._locked():
            base_text = self.load_base_text()
            entries = self.load_entries()
            parts: list[str] = []
            if base_text:
                parts.append(base_text)
            if entries:
                managed_lines = [self.section_title + ":"]
                for entry in entries:
                    managed_lines.append(f"- {entry.key}: {entry.instruction}")
                parts.append("\n".join(managed_lines))
            rendered = "\n\n".join(part for part in parts if part).strip()
            return rendered or None

    def ensure_remote_snapshot(self) -> bool:
        """Seed the remote snapshot when remote-primary storage is enabled.

        Returns:
            ``True`` when a previously missing or invalid remote snapshot was
            created, otherwise ``False``.
        """

        with self._locked():
            if not self._remote_enabled():
                return False

            status, _entries = self._try_load_remote_entries()
            if status == "ok":
                return False
            if status == "error":
                # AUDIT-FIX(#4): Network or backend outages should not crash startup-time snapshot checks.
                return False
            return self._try_save_remote_entries(self._load_local_entries())

    def _split_document(self) -> tuple[str, tuple[ManagedContextEntry, ...], str]:
        text = _read_text_file(self.path)
        if text is None:
            return "", (), ""

        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_text.split("\n")
        start_index = next((index for index, line in enumerate(lines) if line.strip() == _MANAGED_START), None)
        if start_index is None:
            return normalized_text.rstrip(), (), ""

        end_index = next(
            (index for index in range(start_index + 1, len(lines)) if lines[index].strip() == _MANAGED_END),
            None,
        )
        if end_index is None:
            LOGGER.warning("Managed context markers are incomplete in %s; ignoring the managed block.", self.path)
            return normalized_text.rstrip(), (), ""

        # AUDIT-FIX(#5): Parse only exact marker lines so marker text inside content cannot truncate the managed section.
        prefix = "\n".join(lines[:start_index]).rstrip()
        managed_block_lines = lines[start_index + 1 : end_index]
        suffix = "\n".join(lines[end_index + 1 :]).lstrip("\n")

        entries: list[ManagedContextEntry] = []
        for raw_line in managed_block_lines:
            parsed = _parse_markdown_field(raw_line, allow_uppercase_key=True)
            if parsed is None:
                continue
            key, value = parsed
            instruction = _normalize_text(value, limit=220)
            if not instruction:
                continue
            entries.append(
                ManagedContextEntry(
                    key=_slugify(key, fallback="update"),
                    instruction=instruction,
                )
            )
        return prefix, tuple(entries), suffix

    def _write_entries(self, entries: tuple[ManagedContextEntry, ...]) -> None:
        prefix, _existing_entries, suffix = self._split_document()
        self._write_document(prefix=prefix, entries=entries, suffix=suffix)

    def _load_local_entries(self) -> tuple[ManagedContextEntry, ...]:
        _prefix, managed_entries, _suffix = self._split_document()
        return managed_entries

    def _entries_from_payload(self, payload: dict[str, object]) -> tuple[ManagedContextEntry, ...]:
        if payload.get("schema") != _MANAGED_CONTEXT_SCHEMA:
            return ()
        if payload.get("version") != _MANAGED_CONTEXT_VERSION:
            return ()
        items = payload.get("entries")
        if not isinstance(items, list):
            return ()
        entries: list[ManagedContextEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = _slugify(str(item.get("key", "")), fallback="update")
            instruction = _normalize_text(str(item.get("instruction", "")), limit=220)
            if not instruction:
                continue
            entries.append(
                ManagedContextEntry(
                    key=key,
                    instruction=instruction,
                    updated_at=_parse_datetime(str(item.get("updated_at", ""))),
                )
            )
        return tuple(entries)

    def _is_managed_context_payload(self, payload: Mapping[str, object]) -> bool:
        if payload.get("schema") != _MANAGED_CONTEXT_SCHEMA:
            return False
        if payload.get("version") != _MANAGED_CONTEXT_VERSION:
            return False
        return isinstance(payload.get("entries"), list)

    def _try_load_remote_entries(self) -> tuple[str, tuple[ManagedContextEntry, ...]]:
        if not self._remote_enabled():
            return "disabled", ()
        if self.remote_state is None or self.remote_snapshot_kind is None:
            return "disabled", ()
        try:
            payload = self.remote_state.load_snapshot(snapshot_kind=self.remote_snapshot_kind)
        except Exception as exc:
            if _is_remote_unavailable_error(exc):
                raise
            # AUDIT-FIX(#4): Remote state is optional at runtime; backend failures must degrade to local state.
            LOGGER.warning(
                "Failed to load managed-context snapshot %r: %s",
                self.remote_snapshot_kind,
                exc,
            )
            return "error", ()

        if payload is None:
            return "missing", ()
        if not isinstance(payload, dict):
            LOGGER.warning("Managed-context snapshot %r has a non-dict payload.", self.remote_snapshot_kind)
            return "invalid", ()
        if not self._is_managed_context_payload(payload):
            LOGGER.warning("Managed-context snapshot %r has an invalid schema.", self.remote_snapshot_kind)
            return "invalid", ()
        return "ok", self._entries_from_payload(payload)

    def _try_save_remote_entries(self, entries: tuple[ManagedContextEntry, ...]) -> bool:
        if not self._remote_enabled():
            return False
        try:
            self._save_remote_entries(entries)
            return True
        except Exception as exc:
            if _is_remote_unavailable_error(exc):
                raise
            # AUDIT-FIX(#4): Preserve functionality during intermittent connectivity instead of failing the whole update.
            LOGGER.warning(
                "Failed to save managed-context snapshot %r: %s",
                self.remote_snapshot_kind,
                exc,
            )
            return False

    def _persist_entries(self, entries: tuple[ManagedContextEntry, ...]) -> bool:
        if self._remote_enabled():
            return self._try_save_remote_entries(entries)

        try:
            self._write_entries(entries)
            return True
        except (OSError, UnicodeError) as exc:
            LOGGER.warning("Failed to mirror managed context to %s: %s", self.path, exc)
            return False

    def _save_remote_entries(self, entries: tuple[ManagedContextEntry, ...]) -> None:
        if self.remote_state is None or self.remote_snapshot_kind is None:
            raise RuntimeError("Remote managed-context storage is not configured.")
        payload = {
            "schema": _MANAGED_CONTEXT_SCHEMA,
            "version": _MANAGED_CONTEXT_VERSION,
            "entries": [
                {
                    "key": entry.key,
                    "instruction": entry.instruction,
                    "updated_at": _coerce_utc(entry.updated_at).isoformat(),
                }
                for entry in entries
            ],
        }
        self.remote_state.save_snapshot(snapshot_kind=self.remote_snapshot_kind, payload=payload)

    def _write_document(
        self,
        *,
        prefix: str,
        entries: tuple[ManagedContextEntry, ...],
        suffix: str,
    ) -> None:
        body_parts: list[str] = []
        if prefix:
            body_parts.append(prefix.rstrip())
        if entries:
            managed_lines = [
                _MANAGED_START,
                f"## {self.section_title}",
                "_This section is managed by Twinr. Keep entries short and stable._",
            ]
            for entry in entries:
                managed_lines.append(f"- {entry.key}: {entry.instruction}")
            managed_lines.append(_MANAGED_END)
            body_parts.append("\n".join(managed_lines))
        if suffix:
            body_parts.append(suffix.rstrip())
        rendered = "\n\n".join(part for part in body_parts if part).rstrip() + "\n"
        _atomic_write_text(self.path, rendered)


class PersistentMemoryMarkdownStore:
    """Manage Twinr's durable explicit-memory markdown store.

    This store persists only user-approved durable memories, keeps entries
    bounded, and can switch between local markdown persistence and remote-
    primary snapshots without changing the caller interface.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 24,
        remote_state: "LongTermRemoteStateStore | None" = None,
        remote_snapshot_kind: str = "prompt_memory",
        root_dir: str | Path | None = None,
    ) -> None:
        # AUDIT-FIX(#1): Canonicalize the target path before any read/write operation.
        self.path = _resolve_storage_path(path, root_dir=root_dir)
        self.max_entries = _coerce_limit(max_entries, default=_DEFAULT_MAX_ENTRIES)
        self.remote_state = remote_state
        self.remote_snapshot_kind = _normalize_text(remote_snapshot_kind, limit=80) or "prompt_memory"
        # AUDIT-FIX(#5): Use shared named locks to prevent lost updates under concurrent access.
        self._lock_names = [f"path::{self.path}", f"snapshot::{self.remote_snapshot_kind}"]

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _acquire_named_locks(*self._lock_names):
            yield

    def _remote_enabled(self) -> bool:
        return bool(self.remote_state is not None and self.remote_state.enabled)

    def _migration_enabled(self) -> bool:
        config = getattr(self.remote_state, "config", None)
        return bool(config is not None and getattr(config, "long_term_memory_migration_enabled", False))

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        remote_state: "LongTermRemoteStateStore | None" = None,
    ) -> "PersistentMemoryMarkdownStore":
        """Build the durable memory store from Twinr config."""

        from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

        if remote_state is None:
            remote_state = LongTermRemoteStateStore.from_config(config)

        return cls(
            config.memory_markdown_path,
            remote_state=remote_state,
            root_dir=config.project_root,
        )

    def load_entries(self) -> tuple[PersistentMemoryEntry, ...]:
        """Load durable memory entries from remote or local storage."""

        with self._locked():
            if self._remote_enabled():
                # AUDIT-FIX(#2): Valid empty remote memory snapshots must stay empty instead of silently rehydrating local data.
                status, remote_entries = self._try_load_remote_entries()
                if status == "ok":
                    return remote_entries

                local_entries = self._load_local_entries()
                if local_entries and status in {"missing", "invalid"} and self._migration_enabled():
                    self._try_save_remote_entries(local_entries)
                return local_entries
            return self._load_local_entries()

    def ensure_remote_snapshot(self) -> bool:
        """Seed the remote durable-memory snapshot when needed."""

        with self._locked():
            if not self._remote_enabled():
                return False

            status, _entries = self._try_load_remote_entries()
            if status == "ok":
                return False
            if status == "error":
                # AUDIT-FIX(#4): Avoid crashing the app when the remote backend is temporarily unavailable.
                return False
            return self._try_save_remote_entries(self._load_local_entries())

    def _load_local_entries(self) -> tuple[PersistentMemoryEntry, ...]:
        text = _read_text_file(self.path)
        if text is None:
            return ()

        entries: list[PersistentMemoryEntry] = []
        current: dict[str, str] | None = None
        for raw_line in text.splitlines():
            heading = _parse_markdown_heading(raw_line)
            if heading is not None:
                if current is not None:
                    entry = self._entry_from_mapping(current)
                    if entry is not None:
                        entries.append(entry)
                current = {"entry_id": heading}
                continue
            if current is None:
                continue
            parsed = _parse_markdown_field(raw_line)
            if parsed is None:
                continue
            key, value = parsed
            current[key] = value
        if current is not None:
            entry = self._entry_from_mapping(current)
            if entry is not None:
                entries.append(entry)
        return tuple(entries)

    def remember(
        self,
        *,
        kind: str,
        summary: str,
        details: str | None = None,
    ) -> PersistentMemoryEntry:
        """Create or update one durable explicit-memory entry.

        Args:
            kind: Memory category label such as ``contact`` or ``appointment``.
            summary: Short durable summary text.
            details: Optional longer durable detail text.

        Returns:
            The stored ``PersistentMemoryEntry``.

        Raises:
            ValueError: If the summary is empty after normalization.
            RuntimeError: If the entry cannot be persisted.
        """

        clean_kind = _slugify(kind, fallback="memory")
        clean_summary = _normalize_text(summary, limit=220)
        clean_details = _normalize_text(details, limit=420) if details is not None else None
        if not clean_summary:
            raise ValueError("summary must not be empty")

        with self._locked():
            entries = list(self.load_entries())
            now = _utcnow()
            normalized_key = (clean_kind, clean_summary.casefold())
            for index, existing in enumerate(entries):
                if (existing.kind, existing.summary.casefold()) != normalized_key:
                    continue

                # AUDIT-FIX(#8): Distinguish “preserve details” (None) from “explicitly clear details” (blank string).
                updated_details = existing.details if details is None else (clean_details or None)
                updated = PersistentMemoryEntry(
                    entry_id=existing.entry_id,
                    kind=clean_kind,
                    summary=clean_summary,
                    details=updated_details,
                    created_at=existing.created_at,
                    updated_at=now,
                )
                entries[index] = updated
                if not self._persist_entries(tuple(entries)):
                    raise RuntimeError(f"Failed to persist prompt memory update for {self.path}.")
                return updated

            entry = PersistentMemoryEntry(
                entry_id=f"MEM-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
                kind=clean_kind,
                summary=clean_summary,
                details=clean_details or None,
                created_at=now,
                updated_at=now,
            )
            entries.insert(0, entry)
            if len(entries) > self.max_entries:
                entries = entries[: self.max_entries]
            if not self._persist_entries(tuple(entries)):
                raise RuntimeError(f"Failed to persist prompt memory update for {self.path}.")
            return entry

    def delete(self, *, entry_id: str) -> PersistentMemoryEntry | None:
        """Delete one durable explicit-memory entry when it exists."""

        normalized_entry_id = _normalize_entry_id(entry_id)
        with self._locked():
            entries = list(self.load_entries())
            for index, existing in enumerate(entries):
                if existing.entry_id != normalized_entry_id:
                    continue
                removed = entries.pop(index)
                if not self._persist_entries(tuple(entries)):
                    raise RuntimeError(f"Failed to persist prompt memory deletion for {self.path}.")
                return removed
        return None

    def render_context(self, *, limit: int = 12) -> str | None:
        """Render durable memory entries into prompt-context text.

        Args:
            limit: Maximum number of durable entries to include.

        Returns:
            A short durable-memory summary block, or ``None`` when no durable
            entries exist.
        """

        with self._locked():
            entries = self.load_entries()
            if not entries:
                return None
            safe_limit = _coerce_limit(limit, default=_DEFAULT_RENDER_LIMIT)
            lines = ["Durable remembered items explicitly saved for future turns:"]
            for entry in entries[:safe_limit]:
                line = f"- [{entry.kind}] {entry.summary}"
                if entry.details and entry.details.casefold() != entry.summary.casefold():
                    line += f" Details: {entry.details}"
                lines.append(line)
            return "\n".join(lines).strip()

    def _write_entries(self, entries: tuple[PersistentMemoryEntry, ...]) -> None:
        lines = [
            "# Twinr Memory",
            "",
            "This file is managed by Twinr.",
            "It stores durable memories only when the user explicitly asks Twinr to remember something for later.",
            "",
            "## Entries",
        ]
        if not entries:
            lines.extend(["", "_No saved memories yet._"])
        else:
            for entry in entries:
                lines.extend(
                    [
                        "",
                        f"### {entry.entry_id}",
                        f"- kind: {entry.kind}",
                        f"- created_at: {_coerce_utc(entry.created_at).isoformat()}",
                        f"- updated_at: {_coerce_utc(entry.updated_at).isoformat()}",
                        f"- summary: {entry.summary}",
                    ]
                )
                if entry.details:
                    lines.append(f"- details: {entry.details}")
        rendered = "\n".join(lines).rstrip() + "\n"
        _atomic_write_text(self.path, rendered)

    def _entries_from_payload(self, payload: dict[str, object]) -> tuple[PersistentMemoryEntry, ...]:
        if payload.get("schema") != _PROMPT_MEMORY_SCHEMA:
            return ()
        if payload.get("version") != _PROMPT_MEMORY_VERSION:
            return ()
        items = payload.get("entries")
        if not isinstance(items, list):
            return ()
        entries: list[PersistentMemoryEntry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            entry = self._entry_from_mapping(
                {
                    "entry_id": str(item.get("entry_id", "")),
                    "kind": str(item.get("kind", "")),
                    "summary": str(item.get("summary", "")),
                    "details": str(item.get("details", "")) if item.get("details") is not None else "",
                    "created_at": str(item.get("created_at", "")),
                    "updated_at": str(item.get("updated_at", "")),
                }
            )
            if entry is not None:
                entries.append(entry)
        return tuple(entries)

    def _is_prompt_memory_payload(self, payload: Mapping[str, object]) -> bool:
        if payload.get("schema") != _PROMPT_MEMORY_SCHEMA:
            return False
        if payload.get("version") != _PROMPT_MEMORY_VERSION:
            return False
        return isinstance(payload.get("entries"), list)

    def _try_load_remote_entries(self) -> tuple[str, tuple[PersistentMemoryEntry, ...]]:
        if not self._remote_enabled():
            return "disabled", ()
        if self.remote_state is None:
            return "disabled", ()
        try:
            payload = self.remote_state.load_snapshot(snapshot_kind=self.remote_snapshot_kind)
        except Exception as exc:
            if _is_remote_unavailable_error(exc):
                raise
            # AUDIT-FIX(#4): Remote memory backends are optional at runtime; fall back locally on errors.
            LOGGER.warning(
                "Failed to load prompt-memory snapshot %r: %s",
                self.remote_snapshot_kind,
                exc,
            )
            return "error", ()

        if payload is None:
            return "missing", ()
        if not isinstance(payload, dict):
            LOGGER.warning("Prompt-memory snapshot %r has a non-dict payload.", self.remote_snapshot_kind)
            return "invalid", ()
        if not self._is_prompt_memory_payload(payload):
            LOGGER.warning("Prompt-memory snapshot %r has an invalid schema.", self.remote_snapshot_kind)
            return "invalid", ()
        return "ok", self._entries_from_payload(payload)

    def _try_save_remote_entries(self, entries: tuple[PersistentMemoryEntry, ...]) -> bool:
        if not self._remote_enabled():
            return False
        try:
            self._save_remote_entries(entries)
            return True
        except Exception as exc:
            if _is_remote_unavailable_error(exc):
                raise
            # AUDIT-FIX(#4): Do not lose durable memory updates just because the remote backend is flaky.
            LOGGER.warning(
                "Failed to save prompt-memory snapshot %r: %s",
                self.remote_snapshot_kind,
                exc,
            )
            return False

    def _persist_entries(self, entries: tuple[PersistentMemoryEntry, ...]) -> bool:
        if self._remote_enabled():
            return self._try_save_remote_entries(entries)

        try:
            self._write_entries(entries)
            return True
        except (OSError, UnicodeError) as exc:
            LOGGER.warning("Failed to mirror prompt memory to %s: %s", self.path, exc)
            return False

    def _save_remote_entries(self, entries: tuple[PersistentMemoryEntry, ...]) -> None:
        if self.remote_state is None:
            raise RuntimeError("Remote prompt-memory storage is not configured.")
        payload = {
            "schema": _PROMPT_MEMORY_SCHEMA,
            "version": _PROMPT_MEMORY_VERSION,
            "entries": [
                {
                    "entry_id": entry.entry_id,
                    "kind": entry.kind,
                    "summary": entry.summary,
                    "details": entry.details,
                    "created_at": _coerce_utc(entry.created_at).isoformat(),
                    "updated_at": _coerce_utc(entry.updated_at).isoformat(),
                }
                for entry in entries
            ],
        }
        self.remote_state.save_snapshot(snapshot_kind=self.remote_snapshot_kind, payload=payload)

    def _entry_from_mapping(self, data: dict[str, str]) -> PersistentMemoryEntry | None:
        summary = _normalize_text(data.get("summary", ""), limit=220)
        if not summary:
            return None
        return PersistentMemoryEntry(
            entry_id=_normalize_entry_id(data.get("entry_id", "")),
            kind=_slugify(data.get("kind", "memory"), fallback="memory"),
            summary=summary,
            details=_normalize_text(data.get("details", ""), limit=420) or None,
            created_at=_parse_datetime(data.get("created_at", "")),
            updated_at=_parse_datetime(data.get("updated_at", data.get("created_at", ""))),
        )


@dataclass(frozen=True, slots=True)
class PromptContextStore:
    """Group the stores that feed Twinr prompt-context assembly.

    Attributes:
        memory_store: Durable explicit-memory store.
        user_store: Managed user-context store.
        personality_store: Managed personality-context store.
    """

    memory_store: PersistentMemoryMarkdownStore
    user_store: ManagedContextFileStore
    personality_store: ManagedContextFileStore

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "PromptContextStore":
        """Build all prompt-context stores from one Twinr config."""

        from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

        # AUDIT-FIX(#1): Resolve the personality directory under project_root so .env paths cannot escape the project tree.
        personality_dir = _resolve_storage_path(
            config.personality_dir,
            root_dir=config.project_root,
            allow_absolute_outside_root=False,
        )
        remote_state = LongTermRemoteStateStore.from_config(config)
        return cls(
            # AUDIT-FIX(#7): Reuse one LongTermRemoteStateStore so caches/backoff state stay shared on the RPi.
            memory_store=PersistentMemoryMarkdownStore.from_config(config, remote_state=remote_state),
            user_store=ManagedContextFileStore(
                personality_dir / "USER.md",
                section_title="Twinr managed user updates",
                remote_state=remote_state,
                remote_snapshot_kind="user_context",
                root_dir=config.project_root,
            ),
            personality_store=ManagedContextFileStore(
                personality_dir / "PERSONALITY.md",
                section_title="Twinr managed personality updates",
                remote_state=remote_state,
                remote_snapshot_kind="personality_context",
                root_dir=config.project_root,
            ),
        )

    def ensure_remote_snapshots(self) -> tuple[str, ...]:
        """Seed any missing remote snapshots used by prompt-context stores."""

        snapshot_requests = (
            ("prompt_memory", self.memory_store),
            ("user_context", self.user_store),
            ("personality_context", self.personality_store),
        )

        def ensure_one(request: tuple[str, object]) -> tuple[str, bool]:
            default_kind, component = request
            ensure_remote_snapshot = getattr(component, "ensure_remote_snapshot")
            created = bool(ensure_remote_snapshot())
            snapshot_kind = str(getattr(component, "remote_snapshot_kind", "") or default_kind)
            return snapshot_kind, created

        # Keep prompt/user/personality seeding serialized. Live readiness runs
        # share one remote-state/client boundary, and fresh parallel snapshot
        # bootstrap has proven flaky enough to abort required-remote startup.
        results = tuple(ensure_one(request) for request in snapshot_requests)
        return tuple(snapshot_kind for snapshot_kind, created in results if created)
