"""Persist durable long-term memory objects, conflicts, and archives.

This module provides ``LongTermStructuredStore``, the JSON-backed store for
canonical long-term memory snapshots. Import ``LongTermStructuredStore`` from
this module or via ``twinr.memory.longterm``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping  # AUDIT-FIX(#10): Import Mapping explicitly for Python 3.11 type-introspection safety.
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from threading import Lock, RLock
import time
from typing import cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.workflows.forensics import workflow_decision, workflow_event, workflow_span
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
from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogStore
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteReadFailedError,
    LongTermRemoteStateStore,
)
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
_CROSS_SERVICE_READ_MODE = 0o644
_OBJECT_STATE_QUERY_TERMS = frozenset(
    {
        "active",
        "aktuell",
        "available",
        "bestaetigt",
        "candidate",
        "confirmed",
        "confirmed_by_user",
        "current",
        "discarded",
        "expired",
        "former",
        "frueher",
        "gespeichert",
        "invalid",
        "outdated",
        "pending",
        "previous",
        "stored",
        "superseded",
        "unbestaetigt",
        "unclear",
        "uncertain",
        "unklar",
        "unconfirmed",
        "user_confirmed",
        "vorher",
    }
)

_LOG = logging.getLogger(__name__)


def _normalize_text(value: str | None) -> str:
    """Collapse arbitrary text-like input to normalized single-spaced text."""

    return " ".join(str(value or "").split()).strip()


def _retrieval_trace_details(
    query_text: str | None,
    *,
    episodic_limit: int | None = None,
    durable_limit: int | None = None,
    candidate_limit: int | None = None,
    payload_count: int | None = None,
    entry_count: int | None = None,
) -> dict[str, object]:
    """Summarize retrieval inputs for trace-safe latency diagnostics."""

    clean_query = _normalize_text(query_text)
    details: dict[str, object] = {
        "query_chars": len(clean_query),
        "query_terms": len(tuple(term for term in retrieval_terms(clean_query) if isinstance(term, str))),
    }
    if episodic_limit is not None:
        details["episodic_limit"] = max(0, int(episodic_limit))
    if durable_limit is not None:
        details["durable_limit"] = max(0, int(durable_limit))
    if candidate_limit is not None:
        details["candidate_limit"] = max(0, int(candidate_limit))
    if payload_count is not None:
        details["payload_count"] = max(0, int(payload_count))
    if entry_count is not None:
        details["entry_count"] = max(0, int(entry_count))
    return details


def _run_timed_workflow_step(
    *,
    name: str,
    kind: str,
    details: dict[str, object],
    operation: Callable[[], object],
) -> object:
    """Emit bounded timing events for one retrieval step without span rethrow bugs."""

    workflow_event(kind="span_start", msg=name, details={"kind": kind, **details})
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as exc:
        workflow_event(
            kind="exception",
            msg=f"{name}_exception",
            level="ERROR",
            details={
                "span": name,
                "kind": kind,
                "exception": {"type": type(exc).__name__},
            },
            kpi={"duration_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        )
        raise
    workflow_event(
        kind="span_end",
        msg=name,
        details={"kind": kind, **details},
        kpi={"duration_ms": round((time.perf_counter() - started) * 1000.0, 3)},
    )
    return result


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
            os.fchmod(handle.fileno(), _CROSS_SERVICE_READ_MODE)
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
        os.chmod(path, _CROSS_SERVICE_READ_MODE)
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
    _remote_catalog: LongTermRemoteCatalogStore | None = field(init=False, repr=False, default=None)
    _recent_local_snapshot_payloads: dict[str, dict[str, object]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the configured base path once during construction."""

        # AUDIT-FIX(#1): Canonicalize the store root once so subsequent path validation is stable and absolute.
        self.base_path = Path(self.base_path).expanduser().resolve(strict=False)
        self._remote_catalog = LongTermRemoteCatalogStore(self.remote_state)

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
            snapshot_requests = (
                ("objects", self.objects_path, self._empty_objects_payload()),
                ("conflicts", self.conflicts_path, self._empty_conflicts_payload()),
                ("archive", self.archive_path, self._empty_archive_payload()),
            )

            def ensure_one(
                request: tuple[str, Path, dict[str, object]],
            ) -> tuple[str, bool]:
                snapshot_kind, local_path, empty_payload = request
                ensured = self._ensure_remote_snapshot_payload(
                    snapshot_kind=snapshot_kind,
                    local_path=local_path,
                    empty_payload=empty_payload,
                )
                return snapshot_kind, ensured

            # Keep fresh object/conflict/archive bootstrap serialized. Required
            # remote readiness shares one ChonkyDB boundary, and parallel
            # startup seeding has proven flaky enough to trigger bulk-write
            # timeouts on fresh namespaces under backend load.
            results = tuple(ensure_one(request) for request in snapshot_requests)
            return tuple(snapshot_kind for snapshot_kind, ensured in results if ensured)

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
            remote_objects = self.load_objects_by_ids((normalized,))
            if remote_objects:
                return remote_objects[0]
            return next((item for item in self.load_objects() if item.memory_id == normalized), None)

    def load_objects_by_ids(
        self,
        memory_ids: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load a bounded set of long-term objects by memory id when possible."""

        normalized_ids = tuple(
            normalized
            for normalized in (_normalize_text(value) for value in memory_ids)
            if normalized
        )
        if not normalized_ids:
            return ()
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                if remote_catalog.catalog_available(snapshot_kind="objects"):
                    payloads = remote_catalog.load_item_payloads(snapshot_kind="objects", item_ids=normalized_ids)
                    loaded = []
                    for payload in payloads:
                        try:
                            loaded.append(LongTermMemoryObjectV1.from_payload(payload))
                        except Exception:
                            _LOG.warning("Skipping invalid remote long-term object payload during exact load.", exc_info=True)
                    if loaded:
                        by_id = {item.memory_id: item for item in loaded}
                        return tuple(by_id[memory_id] for memory_id in normalized_ids if memory_id in by_id)
            except Exception:
                if self._remote_is_required():
                    raise
                _LOG.warning("Failed loading fine-grained remote long-term objects; falling back to snapshot state.", exc_info=True)
        objects_by_id = {item.memory_id: item for item in self.load_objects()}
        return tuple(objects_by_id[memory_id] for memory_id in normalized_ids if memory_id in objects_by_id)

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
        compatibility_only: bool = False,
    ) -> dict[str, object] | None:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return None
        try:
            raw_payload = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
            remote_catalog = self._remote_catalog
            if (
                snapshot_kind in {"objects", "conflicts", "archive"}
                and remote_catalog is not None
                and remote_catalog.is_catalog_payload(snapshot_kind=snapshot_kind, payload=raw_payload)
            ):
                if compatibility_only:
                    assembly = remote_catalog.assemble_snapshot_from_catalog_result(
                        snapshot_kind=snapshot_kind,
                        payload=raw_payload,
                    )
                    candidate = assembly.payload
                else:
                    candidate = remote_catalog.assemble_snapshot_from_catalog(
                        snapshot_kind=snapshot_kind,
                        payload=raw_payload,
                    )
            elif snapshot_kind in {"objects", "archive"}:
                resolved_payload = self._resolve_sharded_snapshot_payload(
                    snapshot_kind=snapshot_kind,
                    payload=raw_payload,
                )
                candidate = resolved_payload if resolved_payload is not None else raw_payload
                self._maybe_migrate_remote_catalog(
                    snapshot_kind=snapshot_kind,
                    raw_payload=raw_payload,
                    candidate=candidate,
                )
            else:
                candidate = raw_payload
                self._maybe_migrate_remote_catalog(
                    snapshot_kind=snapshot_kind,
                    raw_payload=raw_payload,
                    candidate=candidate,
                )
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
            required_check = cast(Callable[[], object], required)
            try:
                return bool(required_check())  # pylint: disable=not-callable
            except Exception:
                return True
        if required is None:
            return True
        return bool(required)

    def _remote_catalog_enabled(self) -> bool:
        remote_catalog = self._remote_catalog
        return bool(remote_catalog is not None and remote_catalog.enabled())

    def _should_attempt_remote_repair(self) -> bool:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return False
        config = getattr(remote_state, "config", None)
        return bool(getattr(config, "long_term_memory_migration_enabled", False))

    def _maybe_migrate_remote_catalog(
        self,
        *,
        snapshot_kind: str,
        raw_payload: object,
        candidate: object,
    ) -> None:
        remote_catalog = self._remote_catalog
        if (
            snapshot_kind not in {"objects", "conflicts", "archive"}
            or remote_catalog is None
            or remote_catalog.is_catalog_payload(snapshot_kind=snapshot_kind, payload=raw_payload if isinstance(raw_payload, Mapping) else None)
            or not isinstance(candidate, dict)
            or not self._is_valid_snapshot_payload(snapshot_kind=snapshot_kind, payload=candidate)
            or not self._should_attempt_remote_repair()
        ):
            return
        self._persist_remote_snapshot_payload(snapshot_kind=snapshot_kind, payload=candidate)

    def _load_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        local_path: Path,
    ) -> dict[str, object] | None:
        local_path = self._validated_local_path(local_path)
        remote_payload = self._load_remote_snapshot_payload(snapshot_kind=snapshot_kind)
        local_payload = self._read_local_snapshot_payload(snapshot_kind=snapshot_kind, local_path=local_path)
        recent_local_payload = self._same_process_snapshot_bridge_payload(
            snapshot_kind=snapshot_kind,
            remote_payload=remote_payload,
        )

        if self._remote_is_required():
            return recent_local_payload or remote_payload

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

    def _same_process_snapshot_bridge_payload(
        self,
        *,
        snapshot_kind: str,
        remote_payload: Mapping[str, object] | None = None,
    ) -> dict[str, object] | None:
        """Return one fresher local snapshot while remote visibility catches up."""

        if not self._remote_is_required():
            return None
        recent_local_payload = self._recent_local_snapshot_payloads.get(snapshot_kind)
        if recent_local_payload is None:
            return None
        effective_remote_payload = (
            dict(remote_payload)
            if isinstance(remote_payload, Mapping)
            else self._load_remote_snapshot_payload(snapshot_kind=snapshot_kind)
        )
        if effective_remote_payload is None:
            return dict(recent_local_payload)
        if _parse_snapshot_written_at(recent_local_payload) > _parse_snapshot_written_at(effective_remote_payload):
            return dict(recent_local_payload)
        self._recent_local_snapshot_payloads.pop(snapshot_kind, None)
        return None

    def _same_process_snapshot_bridge_objects(self) -> tuple[LongTermMemoryObjectV1, ...] | None:
        """Load the latest same-process object snapshot for bounded query coherence.

        Current-scope `topk_records(scope_ref=...)` can lag behind the already
        visible remote snapshot head because the fine-grained object documents
        and their scope index update asynchronously. Keep query selectors on
        the last in-process object snapshot until a later snapshot read clears
        the bridge after the remote head has caught up.
        """

        if not self._remote_is_required():
            return None
        bridge_payload = self._recent_local_snapshot_payloads.get("objects")
        if bridge_payload is None:
            return None
        return self._load_memory_objects_from_payload(dict(bridge_payload), snapshot_kind="objects")

    def _ensure_remote_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        local_path: Path,
        empty_payload: dict[str, object],
    ) -> bool:
        if self.remote_state is None:
            raise RuntimeError("Remote state store is required to ensure remote snapshots.")  # AUDIT-FIX(#6): Replace assert with a runtime guard that survives python -O.
        local_path = self._validated_local_path(local_path)
        payload = self._load_remote_snapshot_payload(snapshot_kind=snapshot_kind, compatibility_only=True)
        local_payload = self._read_local_snapshot_payload(snapshot_kind=snapshot_kind, local_path=local_path)
        if payload is None:
            payload = self._load_snapshot_payload(snapshot_kind=snapshot_kind, local_path=local_path)
        if payload is None:
            if self._remote_is_required():
                if local_payload is None:
                    self._persist_snapshot_payload(
                        snapshot_kind=snapshot_kind,
                        local_path=local_path,
                        payload=empty_payload,
                    )
                    return True
                self._persist_snapshot_payload(
                    snapshot_kind=snapshot_kind,
                    local_path=local_path,
                    payload=local_payload,
                )
                return True
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
        # AUDIT-FIX(#5): Slot key alone is not unique; use the model-owned compound id for indexing/search.
        return conflict.catalog_item_id()

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
            if snapshot_kind in {"objects", "conflicts", "archive"}:
                remote_catalog = self._remote_catalog
                if remote_catalog is None:
                    raise RuntimeError("Fine-grained remote catalog store is required for structured remote state.")
                remote_payload = {
                    key: value
                    for key, value in payload.items()
                    if key != _SNAPSHOT_WRITTEN_AT_KEY
                }
                catalog_payload = remote_catalog.build_catalog_payload(
                    snapshot_kind=snapshot_kind,
                    item_payloads=self._iter_remote_item_payloads(snapshot_kind=snapshot_kind, payload=remote_payload),
                    item_id_getter=lambda item: self._remote_item_id_for_payload(snapshot_kind=snapshot_kind, payload=item),
                    metadata_builder=lambda item: self._remote_item_metadata(snapshot_kind=snapshot_kind, payload=item),
                    content_builder=lambda item: self._remote_item_search_text(snapshot_kind=snapshot_kind, payload=item),
                )
                if isinstance(payload.get(_SNAPSHOT_WRITTEN_AT_KEY), str):
                    catalog_payload[_SNAPSHOT_WRITTEN_AT_KEY] = payload.get(_SNAPSHOT_WRITTEN_AT_KEY)
                remote_state.save_snapshot(snapshot_kind=snapshot_kind, payload=catalog_payload)
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

    def _iter_remote_item_payloads(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> tuple[dict[str, object], ...]:
        item_key = self._items_key_for_snapshot(snapshot_kind)
        items = payload.get(item_key)
        if not isinstance(items, list):
            return ()
        return tuple(item for item in items if isinstance(item, dict))

    def _remote_item_id_for_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> str | None:
        if snapshot_kind in {"objects", "archive"}:
            return _normalize_text(payload.get("memory_id") if isinstance(payload.get("memory_id"), str) else None)
        if snapshot_kind == "conflicts":
            try:
                conflict = LongTermMemoryConflictV1.from_payload(payload)
            except Exception:
                return None
            return self._conflict_doc_id(conflict)
        raise ValueError(f"Unsupported structured snapshot kind {snapshot_kind!r}.")

    def _remote_item_metadata(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        if snapshot_kind in {"objects", "archive"}:
            metadata: dict[str, object] = {
                "kind": payload.get("kind"),
                "status": payload.get("status"),
                "summary": payload.get("summary"),
                "slot_key": payload.get("slot_key"),
                "value_key": payload.get("value_key"),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }
            if snapshot_kind == "archive":
                metadata["archived_at"] = payload.get("archived_at")
            return metadata
        if snapshot_kind == "conflicts":
            return {
                "slot_key": payload.get("slot_key"),
                "candidate_memory_id": payload.get("candidate_memory_id"),
                "existing_memory_ids": payload.get("existing_memory_ids"),
                "question": payload.get("question"),
                "reason": payload.get("reason"),
                "updated_at": payload.get("question") or payload.get("reason") or payload.get("slot_key"),
            }
        raise ValueError(f"Unsupported structured snapshot kind {snapshot_kind!r}.")

    def _remote_item_search_text(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> str:
        if snapshot_kind in {"objects", "archive"}:
            try:
                return self._object_search_text(LongTermMemoryObjectV1.from_payload(payload))
            except Exception:
                return _normalize_text(json.dumps(dict(payload), ensure_ascii=False))
        if snapshot_kind == "conflicts":
            try:
                conflict = LongTermMemoryConflictV1.from_payload(payload)
            except Exception:
                return _normalize_text(json.dumps(dict(payload), ensure_ascii=False))
            related_parts = [
                conflict.slot_key,
                conflict.question,
                conflict.reason,
                conflict.candidate_memory_id,
                *conflict.existing_memory_ids,
            ]
            return _normalize_text(" ".join(part for part in related_parts if part))
        raise ValueError(f"Unsupported structured snapshot kind {snapshot_kind!r}.")

    def _load_remote_objects_from_entries(
        self,
        *,
        entries: Iterable[object],
        snapshot_kind: str,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        remote_catalog = self._remote_catalog
        if remote_catalog is None:
            return ()
        item_ids = [
            entry.item_id
            for entry in entries
            if hasattr(entry, "item_id") and isinstance(getattr(entry, "item_id"), str)
        ]
        payloads = remote_catalog.load_item_payloads(snapshot_kind=snapshot_kind, item_ids=item_ids)
        loaded: list[LongTermMemoryObjectV1] = []
        for payload in payloads:
            try:
                loaded.append(LongTermMemoryObjectV1.from_payload(payload))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term object payload during selective load.", exc_info=True)
        by_id = {item.memory_id: item for item in loaded}
        return tuple(by_id[item_id] for item_id in item_ids if item_id in by_id)

    def _load_remote_objects_from_payloads(
        self,
        *,
        payloads: Iterable[Mapping[str, object]],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Parse already-materialized remote object payloads."""

        loaded: list[LongTermMemoryObjectV1] = []
        for payload in payloads:
            try:
                loaded.append(LongTermMemoryObjectV1.from_payload(dict(payload)))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term object payload during direct scope search.", exc_info=True)
        return tuple(loaded)

    def _remote_select_objects(
        self,
        *,
        query_text: str | None,
        limit: int,
        include_episodes: bool,
        fallback_limit: int,
        require_query_match: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...] | None:
        remote_catalog = self._remote_catalog
        if not self._remote_catalog_enabled() or remote_catalog is None:
            return None
        bounded_limit = max(1, limit)
        clean_query = _normalize_text(query_text)

        def eligible(entry: object) -> bool:
            metadata = getattr(entry, "metadata", None)
            if not isinstance(metadata, Mapping):
                return False
            kind = _normalize_text(metadata.get("kind") if isinstance(metadata.get("kind"), str) else None)
            status = _normalize_text(metadata.get("status") if isinstance(metadata.get("status"), str) else None)
            if status not in {"active", "candidate", "uncertain"}:
                return False
            if include_episodes:
                return kind == "episode"
            return kind != "episode"

        if not clean_query:
            if require_query_match:
                return ()
            entries = remote_catalog.top_catalog_entries(
                snapshot_kind="objects",
                limit=bounded_limit if fallback_limit <= 0 else min(bounded_limit, max(1, fallback_limit)),
                eligible=eligible,
            )
            return self._load_remote_objects_from_entries(entries=entries, snapshot_kind="objects")

        try:
            direct_payloads = remote_catalog.search_current_item_payloads(
                snapshot_kind="objects",
                query_text=clean_query,
                limit=bounded_limit,
                eligible=eligible,
                allow_catalog_fallback=False,
            )
        except Exception:
            if self._remote_is_required():
                raise
            direct_payloads = None
        if direct_payloads is not None:
            if direct_payloads:
                selected = list(self._load_remote_objects_from_payloads(payloads=direct_payloads))
                filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
                if filtered:
                    return self.rank_selected_objects(
                        query_texts=(clean_query,),
                        objects=filtered,
                        limit=bounded_limit,
                    )
                # Query-driven remote scope searches stay bounded to ranked hits.
                # Falling back to "recent" items here would rehydrate the full
                # remote catalog and reintroduce the 10s+ Pi startup regression.
                return ()
            if include_episodes:
                # Empty episodic scope misses stay authoritative so one off-topic
                # query does not hydrate the full episodic catalog into the turn.
                return ()

        try:
            if not remote_catalog.catalog_available(snapshot_kind="objects"):
                return None
        except Exception:
            if self._remote_is_required():
                raise
            return None

        entries = remote_catalog.search_catalog_entries(
            snapshot_kind="objects",
            query_text=clean_query,
            limit=bounded_limit,
            eligible=eligible,
        )
        selected = list(self._load_remote_objects_from_entries(entries=entries, snapshot_kind="objects"))
        filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
        if filtered:
            return self.rank_selected_objects(
                query_texts=(clean_query,),
                objects=filtered,
                limit=bounded_limit,
            )
        if require_query_match or fallback_limit <= 0:
            return ()
        fallback_entries = remote_catalog.top_catalog_entries(
            snapshot_kind="objects",
            limit=min(bounded_limit, max(1, fallback_limit)),
            eligible=eligible,
        )
        return self._load_remote_objects_from_entries(entries=fallback_entries, snapshot_kind="objects")

    def _select_relevant_objects_from_loaded(
        self,
        *,
        objects: Iterable[LongTermMemoryObjectV1],
        query_text: str | None,
        include_episodes: bool,
        limit: int,
        require_query_match: bool,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Search one already-loaded object pool without re-entering remote scope lookup."""

        bounded_limit = max(1, limit)
        eligible_objects = tuple(
            sorted(
                (
                    item
                    for item in objects
                    if item.status in {"active", "candidate", "uncertain"}
                    and ((item.kind == "episode") if include_episodes else (item.kind != "episode"))
                ),
                key=lambda item: (
                    _coerce_aware_utc(item.updated_at),
                    _coerce_aware_utc(item.created_at),
                    item.memory_id,
                ),
                reverse=True,
            )
        )
        if not eligible_objects:
            return ()
        clean_query = _normalize_text(query_text)
        if not clean_query:
            if require_query_match:
                return ()
            return eligible_objects[:bounded_limit]
        selector = self._object_selector(eligible_objects)
        selected_ids = selector.search(clean_query, limit=bounded_limit)
        by_id = {item.memory_id: item for item in eligible_objects}
        selected = [by_id[memory_id] for memory_id in selected_ids if memory_id in by_id]
        filtered = list(self._filter_query_relevant_objects(clean_query, selected=selected, limit=bounded_limit))
        if not filtered:
            return ()
        return self.rank_selected_objects(
            query_texts=(clean_query,),
            objects=filtered,
            limit=bounded_limit,
        )

    def _remote_select_conflicts(
        self,
        *,
        query_text: str | None,
        limit: int,
    ) -> tuple[LongTermMemoryConflictV1, ...] | None:
        remote_catalog = self._remote_catalog
        if not self._remote_catalog_enabled() or remote_catalog is None:
            return None
        bounded_limit = max(1, limit)
        clean_query = _normalize_text(query_text)
        try:
            if remote_catalog.catalog_item_count(snapshot_kind="conflicts") == 0:
                return ()
        except Exception:
            if self._remote_is_required():
                raise
            return None
        if not clean_query:
            try:
                if not remote_catalog.catalog_available(snapshot_kind="conflicts"):
                    return None
            except Exception:
                if self._remote_is_required():
                    raise
                return None
            entries = remote_catalog.top_catalog_entries(
                snapshot_kind="conflicts",
                limit=bounded_limit,
                preserve_order=True,
            )
            payloads = remote_catalog.load_item_payloads(
                snapshot_kind="conflicts",
                item_ids=(entry.item_id for entry in entries),
            )
        else:
            try:
                direct_payloads = remote_catalog.search_current_item_payloads(
                    snapshot_kind="conflicts",
                    query_text=clean_query,
                    limit=bounded_limit,
                    allow_catalog_fallback=False,
                )
            except Exception:
                if self._remote_is_required():
                    raise
                direct_payloads = None
            if direct_payloads:
                payloads = direct_payloads
            else:
                try:
                    if not remote_catalog.catalog_available(snapshot_kind="conflicts"):
                        return None
                except Exception:
                    if self._remote_is_required():
                        raise
                    return None
                entries = remote_catalog.search_catalog_entries(
                    snapshot_kind="conflicts",
                    query_text=clean_query,
                    limit=bounded_limit,
                )
                payloads = remote_catalog.load_item_payloads(
                    snapshot_kind="conflicts",
                    item_ids=(entry.item_id for entry in entries),
                )
        conflicts: list[LongTermMemoryConflictV1] = []
        for payload in payloads:
            try:
                conflicts.append(LongTermMemoryConflictV1.from_payload(payload))
            except Exception:
                _LOG.warning("Skipping invalid remote long-term conflict payload during selective load.", exc_info=True)
        if not clean_query:
            return tuple(conflicts[:bounded_limit])
        filtered_without_objects = self._filter_query_relevant_conflicts(
            clean_query,
            selected=conflicts,
            limit=bounded_limit,
        )
        if filtered_without_objects or not conflicts:
            return filtered_without_objects
        related_ids = tuple(
            dict.fromkeys(
                memory_id
                for conflict in conflicts
                for memory_id in (conflict.candidate_memory_id, *conflict.existing_memory_ids)
                if isinstance(memory_id, str) and memory_id
            )
        )
        objects_by_id: dict[str, LongTermMemoryObjectV1] = {}
        if related_ids:
            try:
                objects_by_id = {item.memory_id: item for item in self.load_objects_by_ids(related_ids)}
            except Exception:
                if self._remote_is_required():
                    raise
                objects_by_id = {}
        return self._filter_query_relevant_conflicts(
            clean_query,
            selected=conflicts,
            limit=bounded_limit,
            objects_by_id=objects_by_id,
        )

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

        remote_selected = self._remote_select_objects(
            query_text=query_text,
            limit=limit,
            include_episodes=True,
            fallback_limit=fallback_limit,
            require_query_match=require_query_match,
        )
        if remote_selected is not None:
            return remote_selected
        with self._lock:  # AUDIT-FIX(#3): Keep local fallback retrieval consistent with concurrent writes.
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
        self._recent_local_snapshot_payloads[snapshot_kind] = dict(stamped_payload)

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

        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            workflow_event(
                kind="branch",
                msg="longterm_objects_same_process_snapshot_bridge",
                details={
                    "query_present": bool(_normalize_text(query_text)),
                    "limit": max(1, limit),
                },
            )
            return self._select_relevant_objects_from_loaded(
                objects=bridge_objects,
                query_text=query_text,
                include_episodes=False,
                limit=limit,
                require_query_match=False,
            )
        remote_selected = self._remote_select_objects(
            query_text=query_text,
            limit=limit,
            include_episodes=False,
            fallback_limit=0,
            require_query_match=False,
        )
        if remote_selected is not None:
            return remote_selected
        with self._lock:  # AUDIT-FIX(#3): Keep local fallback retrieval consistent with concurrent writes.
            return self._select_relevant_objects_from_loaded(
                objects=self.load_objects(),
                query_text=query_text,
                include_episodes=False,
                limit=limit,
                require_query_match=False,
            )

    def select_fast_topic_objects(
        self,
        *,
        query_text: str | None,
        limit: int = 3,
        timeout_s: float | None = None,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Select a tiny live object set for one bounded fast-topic hint block."""

        bounded_limit = max(1, int(limit))
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return ()
        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            workflow_event(
                kind="branch",
                msg="longterm_fast_topic_same_process_snapshot_bridge",
                details={
                    "query_present": True,
                    "limit": bounded_limit,
                },
            )
            return self._select_fast_topic_objects_from_objects(
                clean_query=clean_query,
                objects=bridge_objects,
                limit=bounded_limit,
            )
        remote_catalog = self._remote_catalog
        if self._remote_catalog_enabled() and remote_catalog is not None:
            try:
                direct_payloads = _run_timed_workflow_step(
                    name="longterm_fast_topic_scope_payload_search",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        durable_limit=bounded_limit,
                        candidate_limit=bounded_limit,
                    ),
                    operation=lambda: remote_catalog.search_current_item_payloads_fast(
                        snapshot_kind="objects",
                        query_text=clean_query,
                        limit=bounded_limit,
                        timeout_s=timeout_s,
                    ),
                )
                return self._filter_fast_topic_objects_for_query(
                    clean_query=clean_query,
                    objects=self._load_remote_objects_from_payloads(payloads=direct_payloads),
                    limit=bounded_limit,
                )
            except LongTermRemoteReadFailedError:
                if self._remote_is_required():
                    raise
                local_selected = self._select_fast_topic_objects_from_local_snapshot(
                    clean_query=clean_query,
                    limit=bounded_limit,
                )
                return local_selected or ()
            except Exception:
                if self._remote_is_required():
                    raise
                local_selected = self._select_fast_topic_objects_from_local_snapshot(
                    clean_query=clean_query,
                    limit=bounded_limit,
                )
                return local_selected or ()
        local_selected = self._select_fast_topic_objects_from_local_snapshot(
            clean_query=clean_query,
            limit=bounded_limit,
        )
        return local_selected or ()

    def _select_fast_topic_objects_from_local_snapshot(
        self,
        *,
        clean_query: str,
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Search the local object snapshot when remote fast-topic retrieval is not required."""

        with self._lock:  # AUDIT-FIX(#3): Keep local snapshot retrieval consistent with concurrent writes.
            return self._select_fast_topic_objects_from_objects(
                clean_query=clean_query,
                objects=self.load_objects(),
                limit=limit,
            )

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

        remote_selected = self._remote_select_conflicts(
            query_text=query_text,
            limit=limit,
        )
        if remote_selected is not None:
            return remote_selected
        with self._lock:  # AUDIT-FIX(#3): Keep local fallback retrieval consistent with concurrent writes.
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
            selected = [by_doc_id[doc_id] for doc_id in selected_doc_ids if doc_id in by_doc_id]
            return self._filter_query_relevant_conflicts(
                clean_query,
                selected=selected,
                limit=bounded_limit,
                objects_by_id=objects_by_id,
            )

    def select_relevant_context_objects(
        self,
        *,
        query_text: str | None,
        episodic_limit: int,
        durable_limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]]:
        """Select episodic and durable objects from one shared query pass.

        Provider-context assembly needs both sections for the same user query.
        Running two independent remote searches makes the Pi pay the same
        ChonkyDB roundtrip twice, so this helper ranks one shared object window
        and then partitions it into episodic and durable sections.
        """

        resolved_episodic_limit = max(0, int(episodic_limit))
        resolved_durable_limit = max(0, int(durable_limit))
        if resolved_episodic_limit <= 0 and resolved_durable_limit <= 0:
            return (), ()

        clean_query = _normalize_text(query_text)
        if not clean_query:
            return (), ()
        bridge_objects = self._same_process_snapshot_bridge_objects()
        if bridge_objects is not None:
            workflow_event(
                kind="branch",
                msg="longterm_context_objects_same_process_snapshot_bridge",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                ),
            )
            return self._rescue_underfilled_context_sections(
                query_text=clean_query,
                partitioned=self._partition_context_objects(
                    query_text=clean_query,
                    objects=bridge_objects,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                ),
                durable_limit=resolved_durable_limit,
            )

        remote_catalog = self._remote_catalog
        shared_limit = self._shared_context_object_limit(
            episodic_limit=resolved_episodic_limit,
            durable_limit=resolved_durable_limit,
        )
        retry_limit = max(resolved_episodic_limit, resolved_durable_limit, 1)
        workflow_decision(
            msg="longterm_context_objects_selection_strategy",
            question="Which retrieval route should build the shared episodic and durable object context?",
            selected={
                "id": "remote_scope_then_catalog" if self._remote_catalog_enabled() and remote_catalog is not None else "local_selector",
                "summary": (
                    "Prefer current-scope remote retrieval, then catalog rescue."
                    if self._remote_catalog_enabled() and remote_catalog is not None
                    else "Use the local selector over already loaded object snapshots."
                ),
            },
            options=[
                {
                    "id": "remote_scope_then_catalog",
                    "summary": "Use ChonkyDB current-scope retrieval first and fall back to catalog-backed selection only when needed.",
                    "score_components": {
                        "remote_catalog_enabled": bool(self._remote_catalog_enabled() and remote_catalog is not None),
                        "shared_limit": shared_limit,
                        "retry_limit": retry_limit,
                    },
                    "constraints_violated": [] if self._remote_catalog_enabled() and remote_catalog is not None else ["remote_catalog_disabled"],
                },
                {
                    "id": "local_selector",
                    "summary": "Search the already loaded local object snapshot directly.",
                    "score_components": {
                        "remote_catalog_enabled": bool(self._remote_catalog_enabled() and remote_catalog is not None),
                        "shared_limit": shared_limit,
                    },
                    "constraints_violated": [] if not (self._remote_catalog_enabled() and remote_catalog is not None) else ["not_preferred_when_remote_catalog_available"],
                },
            ],
            context=_retrieval_trace_details(
                clean_query,
                episodic_limit=resolved_episodic_limit,
                durable_limit=resolved_durable_limit,
            ),
            confidence="high",
            guardrails=[
                "Use one shared object search so the Pi does not pay the same remote roundtrip twice.",
                "Keep current-scope retrieval authoritative before falling back to catalog hydration.",
            ],
            kpi_impact_estimate={
                "shared_limit": shared_limit,
                "retry_limit": retry_limit,
            },
        )
        if self._remote_catalog_enabled() and remote_catalog is not None:
            attempted_limits: list[int] = []
            for candidate_limit in (shared_limit, retry_limit):
                if candidate_limit in attempted_limits:
                    continue
                attempted_limits.append(candidate_limit)
                try:
                    direct_payloads = _run_timed_workflow_step(
                        name="longterm_context_objects_scope_payload_search",
                        kind="retrieval",
                        details=_retrieval_trace_details(
                            clean_query,
                            episodic_limit=resolved_episodic_limit,
                            durable_limit=resolved_durable_limit,
                            candidate_limit=candidate_limit,
                        ),
                        operation=lambda candidate_limit=candidate_limit: remote_catalog.search_current_item_payloads(
                            snapshot_kind="objects",
                            query_text=clean_query,
                            limit=candidate_limit,
                            allow_catalog_fallback=False,
                        ),
                    )
                except Exception:
                    if self._remote_is_required():
                        raise
                    direct_payloads = None
                if direct_payloads is None:
                    workflow_event(
                        kind="branch",
                        msg="longterm_context_objects_scope_unavailable",
                        details=_retrieval_trace_details(
                            clean_query,
                            episodic_limit=resolved_episodic_limit,
                            durable_limit=resolved_durable_limit,
                            candidate_limit=candidate_limit,
                        ),
                        reason={
                            "selected": {"id": "catalog_rescue", "summary": "Current-scope retrieval was unavailable; continue into catalog-backed selection."},
                            "options": [
                                {"id": "catalog_rescue", "summary": "Continue into catalog-backed selection.", "constraints_violated": []},
                                {"id": "abort_scope_retry", "summary": "Abort shared context lookup entirely.", "constraints_violated": ["would_hide_relevant_memory"]},
                            ],
                        },
                    )
                    continue
                with workflow_span(
                    name="longterm_context_objects_partition_shared_pool",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                        candidate_limit=candidate_limit,
                        payload_count=len(direct_payloads),
                    ),
                ):
                    shared_objects = self._load_remote_objects_from_payloads(payloads=direct_payloads)
                    partitioned = self._partition_context_objects(
                        query_text=clean_query,
                        objects=shared_objects,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                    )
                # Current-scope hits can briefly deserialize into payloads that
                # no longer survive the active/candidate/uncertain partition
                # after a fresh confirmation or supersede write. In that case,
                # keep going into catalog-backed rescue instead of returning an
                # empty provider-context durable section.
                if direct_payloads and not partitioned[0] and not partitioned[1]:
                    if not self._collapsed_scope_partition_needs_catalog_rescue(
                        query_text=clean_query,
                        objects=shared_objects,
                    ):
                        workflow_event(
                            kind="branch",
                            msg="longterm_context_objects_scope_authoritative_miss",
                            details=_retrieval_trace_details(
                                clean_query,
                                episodic_limit=resolved_episodic_limit,
                                durable_limit=resolved_durable_limit,
                                candidate_limit=candidate_limit,
                                payload_count=len(direct_payloads),
                            ),
                            reason={
                                "selected": {
                                    "id": "return_empty_partition",
                                    "summary": "Current-scope hits stayed off-topic after active-status filtering, so broader catalog rescue would only repeat the same miss work.",
                                },
                                "options": [
                                    {
                                        "id": "return_empty_partition",
                                        "summary": "Treat the active scope payloads as an authoritative semantic miss.",
                                        "constraints_violated": [],
                                    },
                                    {
                                        "id": "retry_or_catalog_rescue",
                                        "summary": "Continue into the next candidate window or catalog rescue.",
                                        "constraints_violated": ["would_repeat_off_topic_remote_work"],
                                    },
                                ],
                            },
                        )
                        return (), ()
                    workflow_event(
                        kind="branch",
                        msg="longterm_context_objects_scope_partition_empty",
                        details=_retrieval_trace_details(
                            clean_query,
                            episodic_limit=resolved_episodic_limit,
                            durable_limit=resolved_durable_limit,
                            candidate_limit=candidate_limit,
                            payload_count=len(direct_payloads),
                        ),
                        reason={
                            "selected": {"id": "retry_or_catalog_rescue", "summary": "Scope hits collapsed after partitioning; keep searching instead of returning empty context."},
                            "options": [
                                {"id": "retry_or_catalog_rescue", "summary": "Continue into the next candidate window or catalog rescue.", "constraints_violated": []},
                                {"id": "return_empty_partition", "summary": "Return no shared context objects.", "constraints_violated": ["would_hide_fresh_current_facts"]},
                            ],
                        },
                    )
                    continue
                with workflow_span(
                    name="longterm_context_objects_rescue_underfilled_sections",
                    kind="retrieval",
                    details=_retrieval_trace_details(
                        clean_query,
                        episodic_limit=resolved_episodic_limit,
                        durable_limit=resolved_durable_limit,
                        candidate_limit=candidate_limit,
                        payload_count=len(direct_payloads),
                    ),
                ):
                    return self._rescue_underfilled_context_sections(
                        query_text=clean_query,
                        partitioned=partitioned,
                        durable_limit=resolved_durable_limit,
                    )
            try:
                if not remote_catalog.catalog_available(snapshot_kind="objects"):
                    return (), ()
            except Exception:
                if self._remote_is_required():
                    raise
                return (), ()
            entries = _run_timed_workflow_step(
                name="longterm_context_objects_catalog_search",
                kind="retrieval",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                    candidate_limit=shared_limit,
                ),
                operation=lambda: remote_catalog.search_catalog_entries(
                    snapshot_kind="objects",
                    query_text=clean_query,
                    limit=shared_limit,
                ),
            )
            with workflow_span(
                name="longterm_context_objects_catalog_hydrate",
                kind="retrieval",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                    entry_count=len(entries),
                ),
            ):
                shared_objects = self._load_remote_objects_from_entries(
                    entries=entries,
                    snapshot_kind="objects",
                )
                partitioned = self._partition_context_objects(
                    query_text=clean_query,
                    objects=shared_objects,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                )
            with workflow_span(
                name="longterm_context_objects_rescue_underfilled_sections",
                kind="retrieval",
                details=_retrieval_trace_details(
                    clean_query,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                    entry_count=len(entries),
                ),
            ):
                return self._rescue_underfilled_context_sections(
                    query_text=clean_query,
                    partitioned=partitioned,
                    durable_limit=resolved_durable_limit,
                )

        with self._lock:  # AUDIT-FIX(#3): Keep local fallback retrieval consistent with concurrent writes.
            objects = tuple(
                sorted(
                    (
                        item
                        for item in self.load_objects()
                        if item.status in {"active", "candidate", "uncertain"}
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
                return (), ()
            selector = self._object_selector(objects)
            selected_ids = selector.search(clean_query, limit=shared_limit)
            by_id = {item.memory_id: item for item in objects}
            selected = tuple(by_id[memory_id] for memory_id in selected_ids if memory_id in by_id)
            return self._rescue_underfilled_context_sections(
                query_text=clean_query,
                partitioned=self._partition_context_objects(
                    query_text=clean_query,
                    objects=selected,
                    episodic_limit=resolved_episodic_limit,
                    durable_limit=resolved_durable_limit,
                ),
                durable_limit=resolved_durable_limit,
            )

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

    def rank_selected_objects(
        self,
        *,
        query_texts: Iterable[str],
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int | None = None,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Rank selected objects by query overlap, confirmation state, and recency."""

        unique_objects: list[LongTermMemoryObjectV1] = []
        seen_memory_ids: set[str] = set()
        for item in objects:
            if item.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(item.memory_id)
            unique_objects.append(item)
        if not unique_objects:
            return ()
        bounded_limit = max(1, limit) if isinstance(limit, int) else len(unique_objects)
        query_terms = self._combined_query_terms(query_texts)
        ranked = sorted(
            enumerate(unique_objects),
            key=lambda pair: self._object_query_sort_key(
                item=pair[1],
                query_terms=query_terms,
                original_index=pair[0],
            ),
            reverse=True,
        )
        return tuple(item for _index, item in ranked[:bounded_limit])

    def _select_fast_topic_objects_from_loaded(
        self,
        *,
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Keep fast-topic recall tiny and limited to current live objects."""

        selected: list[LongTermMemoryObjectV1] = []
        seen_memory_ids: set[str] = set()
        bounded_limit = max(1, int(limit))
        for item in objects:
            if item.memory_id in seen_memory_ids:
                continue
            if item.status not in {"active", "candidate", "uncertain"}:
                continue
            seen_memory_ids.add(item.memory_id)
            selected.append(item)
            if len(selected) >= bounded_limit:
                break
        return tuple(selected)

    def _filter_fast_topic_objects_for_query(
        self,
        *,
        clean_query: str,
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Keep quick-memory hints topic-anchored for the current query.

        Fast topic reads are intentionally tiny and latency-first, but they
        still need the same content-bearing overlap gate that the wider durable
        recall path already applies. Without that second-stage filter, a weak
        semantic hit such as an old jam preference can leak into control
        questions like ``Was ist ein Regenbogen?`` simply because it was the
        best object available in a tiny current-scope top-k result set.
        """

        eligible_objects = list(self._select_fast_topic_objects_from_loaded(objects=objects, limit=max(1, int(limit))))
        if not eligible_objects:
            return ()
        filtered = self._filter_query_relevant_objects(
            clean_query,
            selected=eligible_objects,
            limit=max(1, int(limit)),
        )
        if not filtered:
            return ()
        return self._select_fast_topic_objects_from_loaded(objects=filtered, limit=max(1, int(limit)))

    def _select_fast_topic_objects_from_objects(
        self,
        *,
        clean_query: str,
        objects: Iterable[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        """Run the bounded fast-topic selector over one already-loaded object pool."""

        eligible_objects = tuple(
            item
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        if not eligible_objects:
            return ()
        selector = self._object_selector(eligible_objects)
        selected_ids = selector.search(clean_query, limit=limit)
        by_id = {item.memory_id: item for item in eligible_objects}
        selected = tuple(by_id[memory_id] for memory_id in selected_ids if memory_id in by_id)
        return self._filter_fast_topic_objects_for_query(
            clean_query=clean_query,
            objects=selected,
            limit=limit,
        )

    def _shared_context_object_limit(
        self,
        *,
        episodic_limit: int,
        durable_limit: int,
    ) -> int:
        """Choose one ranked candidate window for shared context-object search."""

        max_limit = max(episodic_limit, durable_limit, 1)
        return max(12, max_limit * 4)

    def _partition_context_objects(
        self,
        *,
        query_text: str,
        objects: Iterable[LongTermMemoryObjectV1],
        episodic_limit: int,
        durable_limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]]:
        """Split one shared ranked object pool into episodic and durable sections."""

        eligible_objects = tuple(
            item
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        episodic_candidates = [item for item in eligible_objects if item.kind == "episode"]
        durable_candidates = [item for item in eligible_objects if item.kind != "episode"]
        episodic_filtered = self._filter_query_relevant_objects(
            query_text,
            selected=episodic_candidates,
            limit=max(episodic_limit, 1),
        )
        durable_filtered = self._filter_query_relevant_objects(
            query_text,
            selected=durable_candidates,
            limit=max(durable_limit, 1),
        )
        episodic_ranked = (
            self.rank_selected_objects(
                query_texts=(query_text,),
                objects=episodic_filtered,
                limit=episodic_limit,
            )
            if episodic_limit > 0
            else ()
        )
        durable_ranked = (
            self.rank_selected_objects(
                query_texts=(query_text,),
                objects=durable_filtered,
                limit=durable_limit,
            )
            if durable_limit > 0
            else ()
        )
        return episodic_ranked, durable_ranked

    def _collapsed_scope_partition_needs_catalog_rescue(
        self,
        *,
        query_text: str,
        objects: Iterable[LongTermMemoryObjectV1],
    ) -> bool:
        """Return whether an empty scope partition may still hide current facts.

        Current-scope `topk_records(scope_ref=...)` can return stale payloads
        right after a confirmation/supersede write. When the candidate objects
        still overlap the query semantically, the broader catalog path can
        recover the fresh record and must stay available. Pure off-topic drift,
        however, should stop here instead of paying the same remote catalog
        rescue/hydration cost only to return an empty context again.
        """

        eligible_objects = tuple(
            item
            for item in objects
            if item.status in {"active", "candidate", "uncertain"}
        )
        if not eligible_objects:
            return True
        query_terms = self._query_match_terms(retrieval_terms(query_text))
        if not query_terms:
            return False
        semantic_query_terms = self._semantic_query_terms(query_terms)
        for item in eligible_objects:
            if self._has_query_overlap(
                query_terms=semantic_query_terms or query_terms,
                document_terms=retrieval_terms(self._object_semantic_search_text(item)),
            ):
                return True
        return False

    def _rescue_underfilled_context_sections(
        self,
        *,
        query_text: str,
        partitioned: tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]],
        durable_limit: int,
    ) -> tuple[tuple[LongTermMemoryObjectV1, ...], tuple[LongTermMemoryObjectV1, ...]]:
        """Top up durable context when a mixed shared pool starves durable hits.

        Shared episodic/durable search keeps the hot path to one query window,
        but highly active episodic matches can crowd durable multimodal facts
        out of that window. When the durable side comes back underfilled, run
        one bounded durable-only rescue and rerank the merged durable set.
        """

        episodic_ranked, durable_ranked = partitioned
        if durable_limit <= 0 or len(durable_ranked) >= durable_limit:
            return partitioned
        rescue_candidates = self.select_relevant_objects(
            query_text=query_text,
            limit=durable_limit,
        )
        if not rescue_candidates:
            return partitioned
        merged: list[LongTermMemoryObjectV1] = list(durable_ranked)
        seen_memory_ids = {item.memory_id for item in durable_ranked}
        for item in rescue_candidates:
            if item.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(item.memory_id)
            merged.append(item)
        rescued_durable = self.rank_selected_objects(
            query_texts=(query_text,),
            objects=merged,
            limit=durable_limit,
        )
        return episodic_ranked, rescued_durable

    def _object_search_text(self, item: LongTermMemoryObjectV1) -> str:
        parts = [
            item.kind,
            item.summary,
            item.details or "",
            f"status {item.status}",
            self._object_state_search_text(item),
        ]
        # Raw storage identifiers leak dates and internal keys like `2026-03-14`
        # into retrieval. Keep the search text semantic so math/date questions
        # do not match unrelated memories just because a slot key contains `14`.
        for key, value in (item.attributes or {}).items():
            if key in _NON_SEMANTIC_ATTRIBUTE_KEYS:
                continue
            parts.append(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, bool):
                parts.append("true" if value else "false")
            elif isinstance(value, (list, tuple)):
                parts.extend(str(entry) for entry in value if isinstance(entry, str))
        return _normalize_text(" ".join(part for part in parts if part))

    def _object_state_search_text(self, item: LongTermMemoryObjectV1) -> str:
        parts = [item.status]
        if item.status == "active":
            parts.extend(("current", "stored", "available", "aktuell", "gespeichert"))
        elif item.status == "superseded":
            parts.extend(("previous", "former", "superseded", "frueher", "vorher"))
        elif item.status in {"candidate", "uncertain"}:
            parts.extend(("pending", "unconfirmed", "candidate", "unbestaetigt", "unklar"))
        elif item.status == "invalid":
            parts.extend(("invalid", "discarded"))
        elif item.status == "expired":
            parts.extend(("expired", "outdated"))
        if item.confirmed_by_user:
            parts.extend(("confirmed_by_user", "confirmed", "user_confirmed", "bestaetigt"))
        return _normalize_text(" ".join(parts))

    def _combined_query_terms(self, query_texts: Iterable[str]) -> set[str]:
        query_terms: set[str] = set()
        for query_text in query_texts:
            if not isinstance(query_text, str):
                continue
            query_terms.update(retrieval_terms(query_text))
        return query_terms

    def _query_match_terms(self, query_terms: Iterable[str]) -> set[str]:
        """Prefer content-bearing query terms over auxiliary-word-only overlap."""

        normalized = {
            str(term).strip()
            for term in query_terms
            if isinstance(term, str) and str(term).strip()
        }
        if not normalized:
            return set()
        informative = {
            term
            for term in normalized
            if term.isdigit() or len(term) >= 4
        }
        return informative or normalized

    def _semantic_query_terms(self, query_terms: Iterable[str]) -> set[str]:
        """Return topic-bearing terms after removing memory-state-only vocabulary."""

        return {
            term
            for term in self._query_match_terms(query_terms)
            if term not in _OBJECT_STATE_QUERY_TERMS
        }

    def _has_query_overlap(
        self,
        *,
        query_terms: Iterable[str],
        document_terms: Iterable[str],
    ) -> bool:
        """Return whether document terms overlap one query through exact or compound matches."""

        informative_query_terms = self._query_match_terms(query_terms)
        informative_document_terms = self._query_match_terms(document_terms)
        if not informative_query_terms or not informative_document_terms:
            return False
        if informative_query_terms.intersection(informative_document_terms):
            return True
        for query_term in informative_query_terms:
            for document_term in informative_document_terms:
                if query_term in document_term or document_term in query_term:
                    return True
        return False

    def _object_query_overlap_score(
        self,
        *,
        item: LongTermMemoryObjectV1,
        query_terms: set[str],
    ) -> int:
        match_terms = self._query_match_terms(query_terms)
        if not match_terms:
            return 0
        object_terms = set(retrieval_terms(self._object_search_text(item)))
        return len(match_terms.intersection(object_terms))

    def _object_status_priority(self, status: str) -> int:
        if status == "active":
            return 4
        if status == "candidate":
            return 3
        if status == "uncertain":
            return 2
        if status == "superseded":
            return 1
        return 0

    def _object_query_sort_key(
        self,
        *,
        item: LongTermMemoryObjectV1,
        query_terms: set[str],
        original_index: int,
    ) -> tuple[object, ...]:
        return (
            self._object_query_overlap_score(item=item, query_terms=query_terms),
            1 if item.confirmed_by_user else 0,
            self._object_status_priority(item.status),
            _coerce_aware_utc(item.updated_at),
            _coerce_aware_utc(item.created_at),
            item.confidence,
            -original_index,
        )

    def _filter_query_relevant_objects(
        self,
        query_text: str,
        *,
        selected: list[LongTermMemoryObjectV1],
        limit: int,
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        query_terms = self._query_match_terms(retrieval_terms(query_text))
        if not query_terms:
            return tuple(selected[: max(1, limit)])
        semantic_query_terms = self._semantic_query_terms(query_terms)
        topic_terms_by_id = {
            item.memory_id: tuple(retrieval_terms(self._object_semantic_search_text(item)))
            for item in selected
        }
        if semantic_query_terms:
            semantic_matches = [
                item
                for item in selected
                if self._has_query_overlap(
                    query_terms=semantic_query_terms,
                    document_terms=topic_terms_by_id.get(item.memory_id, ()),
                )
            ]
            if semantic_matches:
                anchor_terms: list[str] = []
                for item in semantic_matches:
                    anchor_terms.extend(topic_terms_by_id.get(item.memory_id, ()))
                expanded_matches = [
                    item
                    for item in selected
                    if self._has_query_overlap(
                        query_terms=anchor_terms,
                        document_terms=topic_terms_by_id.get(item.memory_id, ()),
                    )
                ]
                if expanded_matches:
                    return tuple(expanded_matches[: max(1, limit)])
                return tuple(semantic_matches[: max(1, limit)])
        filtered = [
            item
            for item in selected
            if self._has_query_overlap(
                query_terms=query_terms,
                document_terms=retrieval_terms(self._object_search_text(item)),
            )
        ]
        return tuple(filtered[: max(1, limit)])

    def _object_semantic_search_text(self, item: LongTermMemoryObjectV1) -> str:
        """Return one retrieval text stripped of synthetic state-only terms."""

        return _normalize_text(" ".join(part for part in (item.summary, item.details or "") if part))

    def _filter_query_relevant_conflicts(
        self,
        query_text: str,
        *,
        selected: Iterable[LongTermMemoryConflictV1],
        limit: int,
        objects_by_id: Mapping[str, LongTermMemoryObjectV1] | None = None,
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        query_terms = self._query_match_terms(retrieval_terms(query_text))
        selected_conflicts = list(selected)
        if not query_terms:
            return tuple(selected_conflicts[: max(1, limit)])
        related_objects = objects_by_id or {}
        filtered = [
            conflict
            for conflict in selected_conflicts
            if query_terms.intersection(
                retrieval_terms(
                    self._conflict_search_text(
                        conflict,
                        objects_by_id=related_objects,
                    )
                )
            )
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
