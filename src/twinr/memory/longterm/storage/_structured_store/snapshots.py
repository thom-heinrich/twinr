"""Snapshot IO, validation, and remote mirroring for structured memory state."""

# mypy: disable-error-code=attr-defined

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
import os
from pathlib import Path
from typing import cast

from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1

from .shared import (
    _ARCHIVE_STORE_MANIFEST_SCHEMA,
    _ARCHIVE_STORE_SCHEMA,
    _ARCHIVE_STORE_SHARD_SCHEMA,
    _ARCHIVE_STORE_VERSION,
    _CONFLICT_STORE_SCHEMA,
    _CONFLICT_STORE_VERSION,
    _LOG,
    _OBJECT_STORE_MANIFEST_SCHEMA,
    _OBJECT_STORE_SCHEMA,
    _OBJECT_STORE_SHARD_SCHEMA,
    _OBJECT_STORE_VERSION,
    _SNAPSHOT_WRITTEN_AT_KEY,
    _coerce_positive_int,
    _normalize_text,
    _parse_snapshot_written_at,
    _utcnow,
    _write_json_atomic,
)


class StructuredStoreSnapshotMixin:
    """Own snapshot loading, validation, and remote persistence concerns."""

    def ensure_remote_snapshots(self) -> tuple[str, ...]:
        """Bootstrap any missing remote snapshots from local or empty state."""

        with self._lock:
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

            results = tuple(ensure_one(request) for request in snapshot_requests)
            return tuple(snapshot_kind for snapshot_kind, ensured in results if ensured)

    def load_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
        """Load long-term memory objects from the current object snapshot."""

        with self._lock:
            payload = self._load_snapshot_payload(snapshot_kind="objects", local_path=self.objects_path)
            return self._load_memory_objects_from_payload(payload, snapshot_kind="objects")

    def load_conflicts(self) -> tuple[LongTermMemoryConflictV1, ...]:
        """Load unresolved long-term conflicts from the current snapshot."""

        with self._lock:
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

        with self._lock:
            payload = self._load_snapshot_payload(snapshot_kind="archive", local_path=self.archive_path)
            return self._load_memory_objects_from_payload(payload, snapshot_kind="archive")

    def get_object(self, memory_id: str) -> LongTermMemoryObjectV1 | None:
        """Return one stored memory object by canonical memory ID."""

        with self._lock:
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
        candidate = Path(path)
        parent = candidate.parent.resolve(strict=False)
        if parent != self.base_path:
            raise ValueError(f"Structured store path {candidate!s} escapes the configured base path.")
        return candidate

    def _read_local_snapshot_payload(self, *, snapshot_kind: str, local_path: Path) -> dict[str, object] | None:
        local_path = self._validated_local_path(local_path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(local_path, flags)
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
                return bool(required_check())
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
        """Load the latest same-process object snapshot for bounded query coherence."""

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
            raise RuntimeError("Remote state store is required to ensure remote snapshots.")
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
            raw_memory_id = payload.get("memory_id")
            return _normalize_text(raw_memory_id if isinstance(raw_memory_id, str) else None)
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
        _write_json_atomic(local_path, stamped_payload)
        self._persist_remote_snapshot_payload(snapshot_kind=snapshot_kind, payload=stamped_payload)
        self._recent_local_snapshot_payloads[snapshot_kind] = dict(stamped_payload)
