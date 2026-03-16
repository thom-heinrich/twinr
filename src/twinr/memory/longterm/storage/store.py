"""Persist durable long-term memory objects, conflicts, and archives.

This module provides ``LongTermStructuredStore``, the JSON-backed store for
canonical long-term memory snapshots. Import ``LongTermStructuredStore`` from
this module or via ``twinr.memory.longterm``.
"""

from __future__ import annotations

from collections.abc import Mapping  # AUDIT-FIX(#10): Import Mapping explicitly for Python 3.11 type-introspection safety.
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Lock, RLock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryConflictV1,
    LongTermMemoryObjectV1,
    LongTermConflictResolutionV1,
    LongTermMemoryMutationResultV1,
    LongTermMemoryReviewItemV1,
    LongTermMemoryReviewResultV1,
    LongTermRetentionResultV1,
    LongTermReflectionResultV1,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore, LongTermRemoteUnavailableError
from twinr.text_utils import retrieval_terms


_OBJECT_STORE_SCHEMA = "twinr_memory_object_store"
_OBJECT_STORE_VERSION = 1
_OBJECT_STORE_MANIFEST_SCHEMA = "twinr_memory_object_store_manifest"
_OBJECT_STORE_SHARD_SCHEMA = "twinr_memory_object_store_shard"
_CONFLICT_STORE_SCHEMA = "twinr_memory_conflict_store"
_CONFLICT_STORE_VERSION = 1
_ARCHIVE_STORE_SCHEMA = "twinr_memory_archive_store"
_ARCHIVE_STORE_VERSION = 1
_ARCHIVE_STORE_MANIFEST_SCHEMA = "twinr_memory_archive_store_manifest"
_ARCHIVE_STORE_SHARD_SCHEMA = "twinr_memory_archive_store_shard"
_SNAPSHOT_WRITTEN_AT_KEY = "written_at"
_MIN_AWARE_DATETIME = datetime.min.replace(tzinfo=timezone.utc)

_LOG = logging.getLogger(__name__)


def _normalize_text(value: str | None) -> str:
    """Collapse arbitrary text-like input to normalized single-spaced text."""

    return " ".join(str(value or "").split()).strip()


def _utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""

    return datetime.now(timezone.utc)


def _coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce a value to a positive integer or fall back to ``default``."""

    # AUDIT-FIX(#7): Guard integer coercions coming from persisted attributes / env-derived config.
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _coerce_aware_utc(value: object) -> datetime:
    """Normalize a datetime-like value into an aware UTC timestamp."""

    # AUDIT-FIX(#8): Normalize naive datetimes to aware UTC before comparing or sorting.
    if not isinstance(value, datetime):
        return _MIN_AWARE_DATETIME
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_snapshot_written_at(payload: Mapping[str, object]) -> datetime:
    """Parse the stored snapshot write time or return the minimum sentinel."""

    raw_value = payload.get(_SNAPSHOT_WRITTEN_AT_KEY)
    if not isinstance(raw_value, str) or not raw_value:
        return _MIN_AWARE_DATETIME
    try:
        return _coerce_aware_utc(datetime.fromisoformat(raw_value))
    except ValueError:
        return _MIN_AWARE_DATETIME


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry to disk after an atomic file replacement."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


_NON_SEMANTIC_ATTRIBUTE_KEYS = frozenset(
    {
        "support_count",
        "event_names",
    }
)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON object atomically and durably within ``path.parent``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
        _fsync_directory(path.parent)  # AUDIT-FIX(#1): Fsync the parent directory so the atomic rename survives power loss / sudden reboot.
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


@dataclass(slots=True)
class LongTermStructuredStore:
    """Read, mutate, and mirror durable long-term memory snapshots.

    The store owns the file-backed JSON snapshots for active objects,
    unresolved conflicts, and archived objects. Local writes stay
    authoritative and can be mirrored to ``LongTermRemoteStateStore`` when
    remote-primary mode is enabled.
    """

    base_path: Path
    remote_state: LongTermRemoteStateStore | None = None
    _lock: Lock = field(default_factory=RLock, repr=False)  # AUDIT-FIX(#3): Reentrant lock allows read/write serialization without deadlocking nested calls.

    def __post_init__(self) -> None:
        """Normalize the configured base path once during construction."""

        # AUDIT-FIX(#1): Canonicalize the store root once so subsequent path validation is stable and absolute.
        self.base_path = Path(self.base_path).expanduser().resolve(strict=False)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermStructuredStore":
        """Build a structured store rooted at the configured memory path."""

        return cls(
            base_path=chonkydb_data_path(config),
            remote_state=LongTermRemoteStateStore.from_config(config),
        )

    @property
    def objects_path(self) -> Path:
        """Return the local snapshot path for active object state."""

        return self.base_path / "twinr_memory_objects_v1.json"

    @property
    def conflicts_path(self) -> Path:
        """Return the local snapshot path for unresolved conflicts."""

        return self.base_path / "twinr_memory_conflicts_v1.json"

    @property
    def archive_path(self) -> Path:
        """Return the local snapshot path for archived object state."""

        return self.base_path / "twinr_memory_archive_v1.json"

    def ensure_remote_snapshots(self) -> tuple[str, ...]:
        """Bootstrap any missing remote snapshots from local or empty state."""

        with self._lock:  # AUDIT-FIX(#3): Keep snapshot bootstrap serialized with readers/writers.
            if self.remote_state is None or not self.remote_state.enabled:
                return ()
            ensured: list[str] = []
            for snapshot_kind, local_path, empty_payload in (
                ("objects", self.objects_path, self._empty_objects_payload()),
                ("conflicts", self.conflicts_path, self._empty_conflicts_payload()),
                ("archive", self.archive_path, self._empty_archive_payload()),
            ):
                if self._ensure_remote_snapshot_payload(
                    snapshot_kind=snapshot_kind,
                    local_path=local_path,
                    empty_payload=empty_payload,
                ):
                    ensured.append(snapshot_kind)
            return tuple(ensured)

    def load_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load long-term memory objects from the current object snapshot."""

        with self._lock:  # AUDIT-FIX(#3): Serialize reads with multi-file writes for consistent in-process snapshots.
            payload = self._load_snapshot_payload(snapshot_kind="objects", local_path=self.objects_path)
            return self._load_memory_objects_from_payload(payload, snapshot_kind="objects")

    def load_conflicts(self) -> tuple[LongTermMemoryConflictV1, ...]:
        """Load unresolved long-term conflicts from the current snapshot."""

        with self._lock:  # AUDIT-FIX(#3): Serialize reads with multi-file writes for consistent in-process snapshots.
            payload = self._load_snapshot_payload(snapshot_kind="conflicts", local_path=self.conflicts_path)
            if payload is None:
                return ()
            items = payload.get("conflicts", [])
            if not isinstance(items, list):
                return ()
            conflicts: list[LongTermMemoryConflictV1] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                existing_ids = item.get("existing_memory_ids")
                slot_key = _normalize_text(item.get("slot_key") if isinstance(item.get("slot_key"), str) else None)
                candidate_memory_id = _normalize_text(
                    item.get("candidate_memory_id") if isinstance(item.get("candidate_memory_id"), str) else None
                )
                if not slot_key or not candidate_memory_id:
                    _LOG.warning("Skipping invalid long-term memory conflict payload without slot_key/candidate_memory_id.")
                    continue
                conflicts.append(
                    LongTermMemoryConflictV1(
                        slot_key=slot_key,
                        candidate_memory_id=candidate_memory_id,
                        existing_memory_ids=tuple(
                            _normalize_text(value)
                            for value in existing_ids
                            if isinstance(value, str) and _normalize_text(value)
                        )
                        if isinstance(existing_ids, list)
                        else (),
                        question=_normalize_text(item.get("question") if isinstance(item.get("question"), str) else None),
                        reason=_normalize_text(item.get("reason") if isinstance(item.get("reason"), str) else None),
                    )
                )
            return tuple(conflicts)

    def load_archived_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load archived long-term memory objects from the archive snapshot."""

        with self._lock:  # AUDIT-FIX(#3): Serialize reads with multi-file writes for consistent in-process snapshots.
            payload = self._load_snapshot_payload(snapshot_kind="archive", local_path=self.archive_path)
            return self._load_memory_objects_from_payload(payload, snapshot_kind="archive")

    def get_object(self, memory_id: str) -> LongTermMemoryObjectV1 | None:
        """Return one stored memory object by canonical memory ID."""

        with self._lock:  # AUDIT-FIX(#3): Keep object lookup consistent with concurrent mutation writes.
            normalized = _normalize_text(memory_id)
            if not normalized:
                return None
            return next((item for item in self.load_objects() if item.memory_id == normalized), None)

    def _validated_local_path(self, path: Path) -> Path:
        # AUDIT-FIX(#1): Reject any accidental/malicious path escape from the configured store root.
        candidate = Path(path)
        parent = candidate.parent.resolve(strict=False)
        if parent != self.base_path:
            raise ValueError(f"Structured store path {candidate!s} escapes the configured base path.")
        return candidate

    def _read_local_snapshot_payload(self, *, snapshot_kind: str, local_path: Path) -> dict[str, object] | None:
        local_path = self._validated_local_path(local_path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(local_path, flags)  # AUDIT-FIX(#1): O_NOFOLLOW prevents reading attacker-controlled symlink targets.
        except FileNotFoundError:
            return None
        except OSError:
            _LOG.warning("Failed to open local %s snapshot %s securely.", snapshot_kind, local_path, exc_info=True)
            return None

        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except json.JSONDecodeError:
            _LOG.warning("Ignoring corrupt local %s snapshot at %s.", snapshot_kind, local_path, exc_info=True)
            return None
        except OSError:
            _LOG.warning("Failed reading local %s snapshot at %s.", snapshot_kind, local_path, exc_info=True)
            return None

        if not isinstance(loaded, dict):
            _LOG.warning("Ignoring non-object local %s snapshot at %s.", snapshot_kind, local_path)
            return None
        if not self._is_valid_snapshot_payload(snapshot_kind=snapshot_kind, payload=loaded):
            _LOG.warning("Ignoring invalid-schema local %s snapshot at %s.", snapshot_kind, local_path)
            return None
        return dict(loaded)

    def _load_remote_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
    ) -> dict[str, object] | None:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return None
        try:
            if snapshot_kind in {"objects", "archive"}:
                raw_payload = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
                resolved_payload = self._resolve_sharded_snapshot_payload(
                    snapshot_kind=snapshot_kind,
                    payload=raw_payload,
                )
                candidate = resolved_payload if resolved_payload is not None else raw_payload
            else:
                candidate = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
            if candidate is None:
                return None
            if not isinstance(candidate, dict):
                _LOG.warning("Ignoring non-object remote %s snapshot.", snapshot_kind)
                return None
            if not self._is_valid_snapshot_payload(snapshot_kind=snapshot_kind, payload=candidate):
                _LOG.warning("Ignoring invalid-schema remote %s snapshot.", snapshot_kind)
                return None
            return dict(candidate)
        except Exception:
            if self._remote_is_required():
                raise
            _LOG.warning("Failed loading remote %s snapshot; falling back to local state.", snapshot_kind, exc_info=True)
            return None

    def _remote_is_required(self) -> bool:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return False
        required = getattr(remote_state, "required", None)
        if callable(required):
            try:
                return bool(required())
            except Exception:
                return True
        if required is None:
            return True
        return bool(required)

    def _should_attempt_remote_repair(self) -> bool:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return False
        config = getattr(remote_state, "config", None)
        return bool(getattr(config, "long_term_memory_migration_enabled", False))

    def _load_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        local_path: Path,
    ) -> dict[str, object] | None:
        local_path = self._validated_local_path(local_path)
        remote_payload = self._load_remote_snapshot_payload(snapshot_kind=snapshot_kind)
        local_payload = self._read_local_snapshot_payload(snapshot_kind=snapshot_kind, local_path=local_path)

        if self._remote_is_required():
            return remote_payload

        if remote_payload is not None and local_payload is not None:
            local_written_at = _parse_snapshot_written_at(local_payload)
            remote_written_at = _parse_snapshot_written_at(remote_payload)
            if local_written_at > remote_written_at:
                if self._should_attempt_remote_repair():
                    self._persist_remote_snapshot_payload(snapshot_kind=snapshot_kind, payload=local_payload)
                return local_payload
            return remote_payload
        if remote_payload is not None:
            return remote_payload
        if local_payload is not None:
            if self._should_attempt_remote_repair():
                self._persist_remote_snapshot_payload(snapshot_kind=snapshot_kind, payload=local_payload)
            return local_payload
        return None

    def _ensure_remote_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        local_path: Path,
        empty_payload: dict[str, object],
    ) -> bool:
        if self.remote_state is None:
            raise RuntimeError("Remote state store is required to ensure remote snapshots.")  # AUDIT-FIX(#6): Replace assert with a runtime guard that survives python -O.
        payload = self._load_snapshot_payload(snapshot_kind=snapshot_kind, local_path=local_path)
        if payload is None:
            self._persist_snapshot_payload(
                snapshot_kind=snapshot_kind,
                local_path=local_path,
                payload=empty_payload,
            )
            return True
        if self._is_valid_snapshot_payload(snapshot_kind=snapshot_kind, payload=payload):
            return False
        raise ValueError(f"Remote structured snapshot {snapshot_kind!r} has an invalid schema.")

    def _is_valid_snapshot_payload(self, *, snapshot_kind: str, payload: Mapping[str, object]) -> bool:
        if snapshot_kind == "objects":
            return (
                payload.get("schema") == _OBJECT_STORE_SCHEMA
                and payload.get("version") == _OBJECT_STORE_VERSION
                and isinstance(payload.get("objects"), list)
            )
        if snapshot_kind == "conflicts":
            return (
                payload.get("schema") == _CONFLICT_STORE_SCHEMA
                and payload.get("version") == _CONFLICT_STORE_VERSION
                and isinstance(payload.get("conflicts"), list)
            )
        if snapshot_kind == "archive":
            return (
                payload.get("schema") == _ARCHIVE_STORE_SCHEMA
                and payload.get("version") == _ARCHIVE_STORE_VERSION
                and isinstance(payload.get("objects"), list)
            )
        return False

    def _empty_objects_payload(self) -> dict[str, object]:
        return {
            "schema": _OBJECT_STORE_SCHEMA,
            "version": _OBJECT_STORE_VERSION,
            "objects": [],
        }

    def _empty_conflicts_payload(self) -> dict[str, object]:
        return {
            "schema": _CONFLICT_STORE_SCHEMA,
            "version": _CONFLICT_STORE_VERSION,
            "conflicts": [],
        }

    def _empty_archive_payload(self) -> dict[str, object]:
        return {
            "schema": _ARCHIVE_STORE_SCHEMA,
            "version": _ARCHIVE_STORE_VERSION,
            "objects": [],
        }

    def _stamp_snapshot_payload(self, payload: dict[str, object]) -> dict[str, object]:
        stamped = dict(payload)
        stamped[_SNAPSHOT_WRITTEN_AT_KEY] = _utcnow().isoformat()
        return stamped

    def _conflict_key(self, conflict: LongTermMemoryConflictV1) -> tuple[str, str]:
        return (conflict.slot_key, conflict.candidate_memory_id)

    def _conflict_doc_id(self, conflict: LongTermMemoryConflictV1) -> str:
        # AUDIT-FIX(#5): Slot key alone is not unique; include candidate memory id for indexing/search.
        return f"{conflict.slot_key}\x1f{conflict.candidate_memory_id}"

    def _manifest_schema_for_snapshot(self, snapshot_kind: str) -> str:
        if snapshot_kind == "objects":
            return _OBJECT_STORE_MANIFEST_SCHEMA
        if snapshot_kind == "archive":
            return _ARCHIVE_STORE_MANIFEST_SCHEMA
        raise ValueError(f"Unsupported sharded snapshot kind {snapshot_kind!r}.")

    def _shard_schema_for_snapshot(self, snapshot_kind: str) -> str:
        if snapshot_kind == "objects":
            return _OBJECT_STORE_SHARD_SCHEMA
        if snapshot_kind == "archive":
            return _ARCHIVE_STORE_SHARD_SCHEMA
        raise ValueError(f"Unsupported sharded snapshot kind {snapshot_kind!r}.")

    def _items_key_for_snapshot(self, snapshot_kind: str) -> str:
        if snapshot_kind in {"objects", "archive"}:
            return "objects"
        if snapshot_kind == "conflicts":
            return "conflicts"
        raise ValueError(f"Unsupported structured snapshot kind {snapshot_kind!r}.")

    def _resolve_sharded_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        payload: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if payload is None:
            return None
        manifest_schema = self._manifest_schema_for_snapshot(snapshot_kind)
        if payload.get("schema") != manifest_schema:
            return None
        if payload.get("version") != 1:
            raise ValueError(f"Remote {snapshot_kind!r} manifest has an invalid version.")
        shards = payload.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"Remote {snapshot_kind!r} manifest is missing its shard list.")
        item_key = self._items_key_for_snapshot(snapshot_kind)
        shard_schema = self._shard_schema_for_snapshot(snapshot_kind)
        merged: list[dict[str, object]] = []
        remote_state = self.remote_state
        if remote_state is None:
            raise RuntimeError("Remote state store is required to resolve sharded snapshots.")
        for shard_kind in shards:
            if not isinstance(shard_kind, str) or not shard_kind:
                raise ValueError(f"Remote {snapshot_kind!r} manifest contains an invalid shard id.")
            shard_payload = remote_state.load_snapshot(snapshot_kind=shard_kind)
            if not isinstance(shard_payload, dict):
                raise ValueError(f"Remote {snapshot_kind!r} shard {shard_kind!r} is missing.")
            if shard_payload.get("schema") != shard_schema or shard_payload.get("version") != 1:
                raise ValueError(f"Remote {snapshot_kind!r} shard {shard_kind!r} has an invalid schema.")
            items = shard_payload.get(item_key)
            if not isinstance(items, list):
                raise ValueError(f"Remote {snapshot_kind!r} shard {shard_kind!r} has invalid items.")
            merged.extend(item for item in items if isinstance(item, dict))
        if snapshot_kind == "objects":
            return {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": merged,
                _SNAPSHOT_WRITTEN_AT_KEY: payload.get(_SNAPSHOT_WRITTEN_AT_KEY),
            }
        return {
            "schema": _ARCHIVE_STORE_SCHEMA,
            "version": _ARCHIVE_STORE_VERSION,
            "objects": merged,
            _SNAPSHOT_WRITTEN_AT_KEY: payload.get(_SNAPSHOT_WRITTEN_AT_KEY),
        }

    def _save_sharded_snapshot(
        self,
        *,
        snapshot_kind: str,
        payload: dict[str, object],
    ) -> None:
        remote_state = self.remote_state
        if remote_state is None:
            raise RuntimeError("Remote state store is required to save sharded snapshots.")
        item_key = self._items_key_for_snapshot(snapshot_kind)
        items = payload.get(item_key)
        if not isinstance(items, list):
            raise ValueError(f"Structured payload {snapshot_kind!r} is missing its item list.")
        shard_limit = _coerce_positive_int(
            getattr(remote_state.config, "long_term_memory_remote_shard_max_content_chars", 1),
            default=1,
        )
        shard_schema = self._shard_schema_for_snapshot(snapshot_kind)
        manifest_schema = self._manifest_schema_for_snapshot(snapshot_kind)
        shards: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = [*current, item]
            candidate_payload = {"schema": shard_schema, "version": 1, item_key: candidate}
            candidate_size = len(json.dumps(candidate_payload, ensure_ascii=False, separators=(",", ":")))
            if current and candidate_size > shard_limit:
                shards.append(current)
                current = [item]
            else:
                current = candidate
        if current or not shards:
            shards.append(current)

        shard_kinds: list[str] = []
        for index, shard_items in enumerate(shards):
            shard_kind = f"{snapshot_kind}__part_{index:04d}"
            shard_payload = {"schema": shard_schema, "version": 1, item_key: shard_items}
            remote_state.save_snapshot(snapshot_kind=shard_kind, payload=shard_payload)
            shard_kinds.append(shard_kind)
        manifest_payload = {
            "schema": manifest_schema,
            "version": 1,
            "shards": shard_kinds,
        }
        remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=manifest_payload)

    def _persist_remote_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        payload: dict[str, object],
    ) -> None:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return
        try:
            if snapshot_kind in {"objects", "archive"}:
                self._save_sharded_snapshot(snapshot_kind=snapshot_kind, payload=payload)
                return
            remote_payload = {
                key: value
                for key, value in payload.items()
                if key != _SNAPSHOT_WRITTEN_AT_KEY
            }
            remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=remote_payload)
        except Exception:
            if self._remote_is_required():
                raise
            _LOG.warning("Failed persisting remote %s snapshot; keeping local snapshot as source of truth.", snapshot_kind, exc_info=True)

    def select_relevant_episodic_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 4,
        fallback_limit: int = 2,
        require_query_match: bool = False,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Select episodic memories relevant to one retrieval query.

        Args:
            query_text: Free-text retrieval query. Blank queries return recent
                episodes unless ``require_query_match`` is true.
            limit: Maximum number of episodes to return.
            fallback_limit: Maximum number of recent episodes to return when
                ranked query matches are empty and fallback is allowed.
            require_query_match: If true, suppress fallback and return only
                explicit query matches.

        Returns:
            A tuple of episodic objects ordered by selector rank or recency.
        """

        with self._lock:  # AUDIT-FIX(#3): Keep retrieval consistent with concurrent writes.
            objects = tuple(
                sorted(
                    (
                        item
                        for item in self.load_objects()
                        if item.kind == "episode" and item.status in {"active", "candidate", "uncertain"}
                    ),
                    key=lambda item: (
                        _coerce_aware_utc(item.updated_at),
                        _coerce_aware_utc(item.created_at),
                        item.memory_id,
                    ),
                    reverse=True,
                )
            )
            if not objects:
                return ()
            bounded_limit = max(1, limit)
            clean_query = _normalize_text(query_text)
            if not clean_query:
                if require_query_match:
                    return ()
                return objects[:bounded_limit]
            selector = self._object_selector(objects)
            selected_ids = selector.search(
                clean_query,
                limit=bounded_limit,
                category="object",
                allow_fallback=not require_query_match and fallback_limit > 0,
            )
            by_id = {item.memory_id: item for item in objects}
            selected = [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]
            filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
            if not filtered and not require_query_match and fallback_limit > 0:
                return objects[: min(bounded_limit, fallback_limit)]
            return tuple(filtered[:bounded_limit])

    def _persist_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        local_path: Path,
        payload: dict[str, object],
    ) -> None:
        """Persist one validated snapshot locally and mirror it remotely."""

        local_path = self._validated_local_path(local_path)
        if not self._is_valid_snapshot_payload(snapshot_kind=snapshot_kind, payload=payload):
            raise ValueError(f"Refusing to persist invalid structured snapshot {snapshot_kind!r}.")
        stamped_payload = self._stamp_snapshot_payload(payload)
        _write_json_atomic(local_path, stamped_payload)  # AUDIT-FIX(#6): Always persist a local recovery copy before any remote mirror attempt.
        self._persist_remote_snapshot_payload(snapshot_kind=snapshot_kind, payload=stamped_payload)

    def apply_consolidation(self, result: LongTermConsolidationResultV1) -> None:
        """Persist a consolidation result into object and conflict snapshots."""

        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for item in (*result.episodic_objects, *result.durable_objects, *result.deferred_objects):
                existing = existing_objects.get(item.memory_id)
                existing_objects[item.memory_id] = self._merge_object(
                    existing=existing,
                    incoming=item,
                    increment_support=True,
                )
            existing_conflicts = {self._conflict_key(item): item for item in self.load_conflicts()}  # AUDIT-FIX(#5): Preserve multiple conflicts per slot by using a compound key.
            for conflict in result.conflicts:
                existing_conflicts[self._conflict_key(conflict)] = conflict
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(existing_conflicts.values(), key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=objects_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="conflicts",
                local_path=self.conflicts_path,
                payload=conflicts_payload,
            )

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        """Persist reflected objects and summaries into the object snapshot."""

        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for item in (*result.reflected_objects, *result.created_summaries):
                existing_objects[item.memory_id] = self._merge_object(
                    existing=existing_objects.get(item.memory_id),
                    incoming=item,
                    increment_support=False,
                )
            payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=payload,
            )

    def apply_retention(self, result: LongTermRetentionResultV1) -> None:
        """Persist kept objects and archive retained-off objects."""

        with self._lock:
            objects = {item.memory_id: item for item in result.kept_objects}
            archived_objects = {item.memory_id: item for item in self.load_archived_objects()}
            for item in result.archived_objects:
                archived_objects[item.memory_id] = item
            payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(objects.values(), key=lambda row: row.memory_id)],
            }
            archive_payload = {
                "schema": _ARCHIVE_STORE_SCHEMA,
                "version": _ARCHIVE_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(archived_objects.values(), key=lambda row: row.memory_id)],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="archive",
                local_path=self.archive_path,
                payload=archive_payload,
            )

    def apply_conflict_resolution(self, result: LongTermConflictResolutionV1) -> None:
        """Persist updated objects and the remaining conflict queue."""

        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            for item in result.updated_objects:
                existing_objects[item.memory_id] = item
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(result.remaining_conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=objects_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="conflicts",
                local_path=self.conflicts_path,
                payload=conflicts_payload,
            )

    def write_snapshot(
        self,
        *,
        objects: tuple[LongTermMemoryObjectV1, ...],
        conflicts: tuple[LongTermMemoryConflictV1, ...] = (),
        archived_objects: tuple[LongTermMemoryObjectV1, ...] = (),
    ) -> None:
        """Write complete object, conflict, and archive snapshots at once.

        Args:
            objects: Objects to store in the active object snapshot.
            conflicts: Conflicts to store in the conflict snapshot.
            archived_objects: Objects to store in the archive snapshot.
        """

        with self._lock:
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(objects, key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            archive_payload = {
                "schema": _ARCHIVE_STORE_SCHEMA,
                "version": _ARCHIVE_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(archived_objects, key=lambda row: row.memory_id)],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=objects_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="conflicts",
                local_path=self.conflicts_path,
                payload=conflicts_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="archive",
                local_path=self.archive_path,
                payload=archive_payload,
            )

    def apply_memory_mutation(self, result: LongTermMemoryMutationResultV1) -> None:
        """Persist a user-driven mutation result across all snapshots."""

        with self._lock:
            existing_objects = {item.memory_id: item for item in self.load_objects()}
            archived_objects = {item.memory_id: item for item in self.load_archived_objects()}
            for memory_id in result.deleted_memory_ids:
                existing_objects.pop(memory_id, None)
                archived_objects.pop(memory_id, None)
            for item in result.updated_objects:
                existing_objects[item.memory_id] = item
                archived_objects.pop(item.memory_id, None)
            objects_payload = {
                "schema": _OBJECT_STORE_SCHEMA,
                "version": _OBJECT_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(existing_objects.values(), key=lambda row: row.memory_id)],
            }
            conflicts_payload = {
                "schema": _CONFLICT_STORE_SCHEMA,
                "version": _CONFLICT_STORE_VERSION,
                "conflicts": [
                    item.to_payload()
                    for item in sorted(result.remaining_conflicts, key=lambda row: (row.slot_key, row.candidate_memory_id))
                ],
            }
            archive_payload = {
                "schema": _ARCHIVE_STORE_SCHEMA,
                "version": _ARCHIVE_STORE_VERSION,
                "objects": [item.to_payload() for item in sorted(archived_objects.values(), key=lambda row: row.memory_id)],
            }
            self._persist_snapshot_payload(
                snapshot_kind="objects",
                local_path=self.objects_path,
                payload=objects_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="conflicts",
                local_path=self.conflicts_path,
                payload=conflicts_payload,
            )
            self._persist_snapshot_payload(
                snapshot_kind="archive",
                local_path=self.archive_path,
                payload=archive_payload,
            )

    def review_objects(
        self,
        *,
        query_text: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        include_episodes: bool = False,
        limit: int = 12,
    ) -> LongTermMemoryReviewResultV1:
        """Build a bounded review page over stored memory objects.

        Args:
            query_text: Optional free-text filter over stored objects.
            status: Optional status filter such as ``active`` or ``candidate``.
            kind: Optional memory-kind filter.
            include_episodes: If true, include episodic objects in the review.
            limit: Maximum number of review items to return.

        Returns:
            A review result containing the selected items plus the total match
            count for the applied filters.
        """

        with self._lock:  # AUDIT-FIX(#3): Keep review output consistent with concurrent mutation/write activity.
            bounded_limit = max(1, limit)  # AUDIT-FIX(#9): Normalize nonsensical limits to a safe minimum.
            objects = [
                item
                for item in self.load_objects()
                if (include_episodes or item.kind != "episode")
                and (status is None or item.status == status)
                and (kind is None or item.kind == kind)
            ]
            if not objects:
                return LongTermMemoryReviewResultV1(
                    items=(),
                    total_count=0,
                    query_text=query_text,
                    status_filter=status,
                    kind_filter=kind,
                    include_episodes=include_episodes,
                )
            query_text = _normalize_text(query_text)
            if query_text:
                selector = self._object_selector(objects)
                all_selected_ids = selector.search(query_text, limit=len(objects))  # AUDIT-FIX(#9): Compute the true filtered count instead of a page-sized count.
                selected_by_id = {item.memory_id: item for item in objects}
                ordered_selected = [selected_by_id[memory_id] for memory_id in all_selected_ids if memory_id in selected_by_id]
                selected = ordered_selected[:bounded_limit]
                total_count = len(ordered_selected)
            else:
                selected = sorted(
                    objects,
                    key=lambda item: (_coerce_aware_utc(item.updated_at), item.memory_id),
                    reverse=True,
                )[:bounded_limit]
                total_count = len(objects)
            return LongTermMemoryReviewResultV1(
                items=tuple(self._to_review_item(item) for item in selected),
                total_count=total_count,
                query_text=query_text,
                status_filter=status,
                kind_filter=kind,
                include_episodes=include_episodes,
            )

    def confirm_object(self, memory_id: str) -> LongTermMemoryMutationResultV1:
        """Build a mutation result that confirms one stored object.

        Args:
            memory_id: Canonical ID of the object to confirm.

        Returns:
            A mutation result that marks the object active and user-confirmed.

        Raises:
            ValueError: If no stored object exists for ``memory_id``.
        """

        with self._lock:  # AUDIT-FIX(#3): Build mutation results from a single consistent in-memory view.
            current = self.get_object(memory_id)
            if current is None:
                raise ValueError(f"No long-term memory object found for {memory_id!r}.")
            current_time = _utcnow()
            attrs = dict(current.attributes or {})
            attrs["review_confirmed_by_user"] = True
            attrs["review_confirmed_at"] = current_time.isoformat()
            updated = current.with_updates(
                status="active",
                confirmed_by_user=True,
                confidence=max(current.confidence, 0.99),
                updated_at=current_time,
                attributes=attrs,
            )
            return LongTermMemoryMutationResultV1(
                action="confirm",
                target_memory_id=current.memory_id,
                updated_objects=(updated,),
                remaining_conflicts=self.load_conflicts(),
            )

    def invalidate_object(
        self,
        memory_id: str,
        *,
        reason: str | None = None,
    ) -> LongTermMemoryMutationResultV1:
        """Build a mutation result that invalidates one stored object.

        Args:
            memory_id: Canonical ID of the object to invalidate.
            reason: Optional user-facing reason recorded in object attributes.

        Returns:
            A mutation result containing the invalidated object plus any
            related reference cleanup updates.

        Raises:
            ValueError: If no stored object exists for ``memory_id``.
        """

        with self._lock:  # AUDIT-FIX(#3): Build mutation results from a single consistent in-memory view.
            current = self.get_object(memory_id)
            if current is None:
                raise ValueError(f"No long-term memory object found for {memory_id!r}.")
            current_time = _utcnow()
            attrs = dict(current.attributes or {})
            attrs["invalidated_by_user"] = True
            if reason:
                attrs["invalidation_reason"] = _normalize_text(reason)
            updated_target = current.with_updates(
                status="invalid",
                confirmed_by_user=True,
                conflicts_with=(),
                updated_at=current_time,
                attributes=attrs,
            )
            related_updates = self._cleanup_references_after_mutation(
                target_memory_id=current.memory_id,
                drop_supersedes=False,
            )
            remaining_conflicts = self._rewrite_conflicts_without_memory(current.memory_id)
            return LongTermMemoryMutationResultV1(
                action="invalidate",
                target_memory_id=current.memory_id,
                updated_objects=tuple(
                    sorted((updated_target, *related_updates), key=lambda item: item.memory_id)
                ),
                remaining_conflicts=remaining_conflicts,
            )

    def delete_object(self, memory_id: str) -> LongTermMemoryMutationResultV1:
        """Build a mutation result that removes one stored object.

        Args:
            memory_id: Canonical ID of the object to delete.

        Returns:
            A mutation result describing the deletion and related cleanup.

        Raises:
            ValueError: If no stored object exists for ``memory_id``.
        """

        with self._lock:  # AUDIT-FIX(#3): Build mutation results from a single consistent in-memory view.
            current = self.get_object(memory_id)
            if current is None:
                raise ValueError(f"No long-term memory object found for {memory_id!r}.")
            related_updates = self._cleanup_references_after_mutation(
                target_memory_id=current.memory_id,
                drop_supersedes=True,
            )
            remaining_conflicts = self._rewrite_conflicts_without_memory(current.memory_id)
            return LongTermMemoryMutationResultV1(
                action="delete",
                target_memory_id=current.memory_id,
                updated_objects=tuple(sorted(related_updates, key=lambda item: item.memory_id)),
                deleted_memory_ids=(current.memory_id,),
                remaining_conflicts=remaining_conflicts,
            )

    def select_relevant_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 4,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Select durable non-episodic objects relevant to a query.

        Args:
            query_text: Free-text retrieval query. Blank queries return the
                most recent eligible objects.
            limit: Maximum number of objects to return.

        Returns:
            A tuple of active, candidate, or uncertain non-episodic objects.
        """

        with self._lock:  # AUDIT-FIX(#3): Keep retrieval consistent with concurrent writes.
            bounded_limit = max(1, limit)  # AUDIT-FIX(#9): Prevent negative/zero slicing quirks.
            objects = tuple(
                sorted(
                    (
                        item
                        for item in self.load_objects()
                        if item.kind != "episode" and item.status in {"active", "candidate", "uncertain"}
                    ),
                    key=lambda item: (
                        _coerce_aware_utc(item.updated_at),
                        _coerce_aware_utc(item.created_at),
                        item.memory_id,
                    ),
                    reverse=True,
                )
            )
            if not objects:
                return ()
            clean_query = _normalize_text(query_text)
            if not clean_query:
                return objects[:bounded_limit]
            selector = self._object_selector(objects)
            selected_ids = selector.search(clean_query, limit=bounded_limit)
            by_id = {item.memory_id: item for item in objects}
            selected = [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]
            return tuple(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))

    def select_open_conflicts(
        self,
        *,
        query_text: str | None,
        limit: int = 3,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        """Select unresolved conflicts relevant to a query.

        Args:
            query_text: Free-text retrieval query. Blank queries return the
                earliest conflicts already loaded in storage order.
            limit: Maximum number of conflicts to return.

        Returns:
            A tuple of unresolved conflict records.
        """

        with self._lock:  # AUDIT-FIX(#3): Keep retrieval consistent with concurrent writes.
            conflicts = self.load_conflicts()
            if not conflicts:
                return ()
            bounded_limit = max(1, limit)  # AUDIT-FIX(#9): Prevent negative/zero slicing quirks.
            clean_query = _normalize_text(query_text)
            if not clean_query:
                return conflicts[:bounded_limit]
            objects_by_id = {item.memory_id: item for item in self.load_objects()}
            selector = FullTextSelector(
                tuple(
                    FullTextDocument(
                        doc_id=self._conflict_doc_id(conflict),
                        category="conflict",
                        content=self._conflict_search_text(conflict, objects_by_id=objects_by_id),
                    )
                    for conflict in conflicts
                )
            )
            selected_doc_ids = selector.search(clean_query, limit=bounded_limit, category="conflict")
            by_doc_id = {self._conflict_doc_id(item): item for item in conflicts}
            return tuple(by_doc_id[doc_id] for doc_id in selected_doc_ids if doc_id in by_doc_id)

    def _load_memory_objects_from_payload(
        self,
        payload: dict[str, object] | None,
        *,
        snapshot_kind: str,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        if payload is None:
            return ()
        items = payload.get("objects", [])
        if not isinstance(items, list):
            return ()
        objects: list[LongTermMemoryObjectV1] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                objects.append(LongTermMemoryObjectV1.from_payload(item))
            except Exception:
                _LOG.warning("Skipping invalid long-term memory object in %s snapshot.", snapshot_kind, exc_info=True)  # AUDIT-FIX(#4): Isolate single-record corruption instead of failing the whole load.
        return tuple(objects)

    def _merge_object(
        self,
        *,
        existing: LongTermMemoryObjectV1 | None,
        incoming: LongTermMemoryObjectV1,
        increment_support: bool,
    ) -> LongTermMemoryObjectV1:
        if existing is None:
            attrs = dict(incoming.attributes or {})
            attrs["support_count"] = _coerce_positive_int(attrs.get("support_count", 1), default=1)
            return incoming.with_updates(attributes=attrs)
        attrs = dict(existing.attributes or {})
        incoming_attrs = dict(incoming.attributes or {})
        existing_support = _coerce_positive_int(attrs.get("support_count", 1), default=1)
        incoming_support = _coerce_positive_int(incoming_attrs.get("support_count", 1), default=1)
        support_count = existing_support + 1 if increment_support else max(existing_support, incoming_support)
        attrs.update(incoming_attrs)
        attrs["support_count"] = support_count

        # AUDIT-FIX(#7): Merge source payloads defensively and preserve incoming provenance fields instead of dropping them.
        existing_source_payload = (
            existing.source.to_payload()
            if getattr(existing, "source", None) is not None and hasattr(existing.source, "to_payload")
            else {}
        )
        incoming_source_payload = (
            incoming.source.to_payload()
            if getattr(incoming, "source", None) is not None and hasattr(incoming.source, "to_payload")
            else {}
        )
        existing_event_ids = tuple(
            value for value in existing_source_payload.get("event_ids", []) if isinstance(value, str) and value
        )
        incoming_event_ids = tuple(
            value for value in incoming_source_payload.get("event_ids", []) if isinstance(value, str) and value
        )
        merged_event_ids = tuple(dict.fromkeys((*existing_event_ids, *incoming_event_ids)))
        merged_source = dict(existing_source_payload)
        merged_source.update(incoming_source_payload)
        merged_source["event_ids"] = list(merged_event_ids)

        return incoming.with_updates(
            source=merged_source,
            attributes=attrs,
            created_at=min(_coerce_aware_utc(existing.created_at), _coerce_aware_utc(incoming.created_at)),
            updated_at=max(_coerce_aware_utc(existing.updated_at), _coerce_aware_utc(incoming.updated_at)),
            confidence=max(existing.confidence, incoming.confidence),
            status=self._preferred_status(existing.status, incoming.status),
        )

    def _preferred_status(self, existing: str, incoming: str) -> str:
        rank = {
            "active": 5,
            "candidate": 4,
            "uncertain": 3,
            "superseded": 2,
            "expired": 1,
            "invalid": 0,
        }
        return existing if rank.get(existing, -1) >= rank.get(incoming, -1) else incoming

    def _rewrite_conflicts_without_memory(
        self,
        memory_id: str,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        rewritten: list[LongTermMemoryConflictV1] = []
        for conflict in self.load_conflicts():
            if conflict.candidate_memory_id == memory_id:
                continue
            existing_ids = tuple(value for value in conflict.existing_memory_ids if value != memory_id)
            if not existing_ids:
                continue
            rewritten.append(
                LongTermMemoryConflictV1(
                    slot_key=conflict.slot_key,
                    candidate_memory_id=conflict.candidate_memory_id,
                    existing_memory_ids=existing_ids,
                    question=conflict.question,
                    reason=conflict.reason,
                )
            )
        return tuple(rewritten)

    def _cleanup_references_after_mutation(
        self,
        *,
        target_memory_id: str,
        drop_supersedes: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        updated_objects: list[LongTermMemoryObjectV1] = []
        update_time = _utcnow()
        for item in self.load_objects():
            if item.memory_id == target_memory_id:
                continue
            conflicts_with = tuple(value for value in item.conflicts_with if value != target_memory_id)
            supersedes = (
                tuple(value for value in item.supersedes if value != target_memory_id)
                if drop_supersedes
                else item.supersedes
            )
            if conflicts_with == item.conflicts_with and supersedes == item.supersedes:
                continue
            updated_objects.append(
                item.with_updates(
                    conflicts_with=conflicts_with,
                    supersedes=supersedes,
                    updated_at=update_time,
                )
            )
        return tuple(updated_objects)

    def _to_review_item(self, item: LongTermMemoryObjectV1) -> LongTermMemoryReviewItemV1:
        return LongTermMemoryReviewItemV1(
            memory_id=item.memory_id,
            kind=item.kind,
            summary=item.summary,
            details=item.details,
            status=item.status,
            confidence=item.confidence,
            updated_at=item.updated_at,
            confirmed_by_user=item.confirmed_by_user,
            sensitivity=item.sensitivity,
            slot_key=item.slot_key,
            value_key=item.value_key,
        )

    def _object_selector(
        self,
        objects: tuple[LongTermMemoryObjectV1, ...] | list[LongTermMemoryObjectV1],
    ) -> FullTextSelector:
        return FullTextSelector(
            tuple(
                FullTextDocument(
                    doc_id=item.memory_id,
                    category="object",
                    content=self._object_search_text(item),
                )
                for item in objects
            )
        )

    def _object_search_text(self, item: LongTermMemoryObjectV1) -> str:
        parts = [
            item.kind,
            item.summary,
            item.details or "",
        ]
        for key, value in (item.attributes or {}).items():
            if key in _NON_SEMANTIC_ATTRIBUTE_KEYS:
                continue
            parts.append(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(entry) for entry in value if isinstance(entry, str))
        return _normalize_text(" ".join(part for part in parts if part))

    def _filter_query_relevant_objects(
        self,
        query_text: str,
        *,
        selected: list[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        query_terms = set(retrieval_terms(query_text))
        semantic_query_terms = {term for term in query_terms if not term.isdigit()}
        if not semantic_query_terms:
            return tuple(selected[: max(1, limit)])
        filtered = [
            item
            for item in selected
            if semantic_query_terms.intersection(retrieval_terms(self._object_search_text(item)))
        ]
        return tuple(filtered[: max(1, limit)])

    def _conflict_search_text(
        self,
        conflict: LongTermMemoryConflictV1,
        *,
        objects_by_id: dict[str, LongTermMemoryObjectV1],
    ) -> str:
        related_parts = [
            conflict.slot_key,
            conflict.question,
            conflict.reason,
        ]
        candidate = objects_by_id.get(conflict.candidate_memory_id)
        if candidate is not None:
            related_parts.append(self._object_search_text(candidate))
        for memory_id in conflict.existing_memory_ids:
            existing = objects_by_id.get(memory_id)
            if existing is not None:
                related_parts.append(self._object_search_text(existing))
        return _normalize_text(" ".join(part for part in related_parts if part))


__all__ = ["LongTermStructuredStore"]
