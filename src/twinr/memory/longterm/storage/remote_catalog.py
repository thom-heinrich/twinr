"""Persist fine-grained remote long-term memory documents plus small catalogs.

This module replaces the old remote manifest+shard pattern for long-term
objects, conflicts, and archives. Each memory item is stored as its own
ChonkyDB document, while the existing remote-state snapshot path keeps only
the small current catalog that maps stable Twinr ids to the active ChonkyDB
document ids.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import logging
from urllib.parse import quote

from twinr.memory.chonkydb.models import ChonkyDBBulkRecordRequest, ChonkyDBRecordItem, ChonkyDBRetrieveRequest
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore, LongTermRemoteUnavailableError


_LOG = logging.getLogger(__name__)

_CATALOG_VERSION = 2
_ITEM_VERSION = 1
_DEFAULT_MAX_ITEM_CONTENT_CHARS = 256_000
_DEFAULT_BULK_BATCH_SIZE = 64


@dataclass(frozen=True, slots=True)
class LongTermRemoteCatalogEntry:
    """Describe one current fine-grained remote memory item."""

    snapshot_kind: str
    item_id: str
    document_id: str | None
    uri: str
    metadata: dict[str, object]

    def updated_at(self) -> str:
        """Return the best available update timestamp string for sorting."""

        value = self.metadata.get("updated_at")
        return str(value).strip() if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class _RemoteCollectionDefinition:
    snapshot_kind: str
    catalog_schema: str
    item_schema: str
    envelope_key: str
    uri_segment: str


_DEFINITIONS: dict[str, _RemoteCollectionDefinition] = {
    "objects": _RemoteCollectionDefinition(
        snapshot_kind="objects",
        catalog_schema="twinr_memory_object_catalog_v2",
        item_schema="twinr_memory_object_record_v2",
        envelope_key="object",
        uri_segment="objects",
    ),
    "conflicts": _RemoteCollectionDefinition(
        snapshot_kind="conflicts",
        catalog_schema="twinr_memory_conflict_catalog_v2",
        item_schema="twinr_memory_conflict_record_v2",
        envelope_key="conflict",
        uri_segment="conflicts",
    ),
    "archive": _RemoteCollectionDefinition(
        snapshot_kind="archive",
        catalog_schema="twinr_memory_archive_catalog_v2",
        item_schema="twinr_memory_archive_record_v2",
        envelope_key="object",
        uri_segment="archive",
    ),
}

class LongTermRemoteCatalogStore:
    """Store and query remote long-term memory items as individual documents."""

    def __init__(self, remote_state: LongTermRemoteStateStore | None) -> None:
        self.remote_state = remote_state

    def enabled(self) -> bool:
        """Return whether fine-grained remote storage is available."""

        return bool(self.remote_state is not None and self.remote_state.enabled)

    def is_catalog_payload(self, *, snapshot_kind: str, payload: Mapping[str, object] | None) -> bool:
        """Return whether one remote snapshot payload is a v2 item catalog."""

        definition = self._definition(snapshot_kind)
        if definition is None or not isinstance(payload, Mapping):
            return False
        return (
            payload.get("schema") == definition.catalog_schema
            and payload.get("version") == _CATALOG_VERSION
            and isinstance(payload.get("items"), list)
        )

    def load_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None = None,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Load and normalize the current catalog entries for one collection."""

        definition = self._require_definition(snapshot_kind)
        if payload is None:
            payload = self._load_catalog_payload(snapshot_kind=snapshot_kind)
        if not self.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload):
            return ()
        items = payload.get("items")
        assert isinstance(items, list)
        entries: list[LongTermRemoteCatalogEntry] = []
        for raw_entry in items:
            if not isinstance(raw_entry, Mapping):
                continue
            item_id = self._normalize_text(raw_entry.get("item_id"))
            uri = self._normalize_text(raw_entry.get("uri"))
            if not item_id or not uri:
                continue
            metadata = raw_entry.get("metadata")
            entries.append(
                LongTermRemoteCatalogEntry(
                    snapshot_kind=definition.snapshot_kind,
                    item_id=item_id,
                    document_id=self._normalize_text(raw_entry.get("document_id")),
                    uri=uri,
                    metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                )
            )
        return tuple(entries)

    def catalog_available(self, *, snapshot_kind: str) -> bool:
        """Return whether the current remote snapshot is already a fine-grained catalog."""

        return self.is_catalog_payload(
            snapshot_kind=snapshot_kind,
            payload=self._load_catalog_payload(snapshot_kind=snapshot_kind),
        )

    def assemble_snapshot_from_catalog(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object] | None,
    ) -> dict[str, object] | None:
        """Load every current remote item referenced by one catalog payload."""

        definition = self._require_definition(snapshot_kind)
        entries = self.load_catalog_entries(snapshot_kind=snapshot_kind, payload=payload)
        item_payloads: list[dict[str, object]] = []
        for entry in entries:
            loaded = self.load_item_payload(
                snapshot_kind=definition.snapshot_kind,
                item_id=entry.item_id,
                document_id=entry.document_id,
                uri=entry.uri,
            )
            if loaded is None:
                raise LongTermRemoteUnavailableError(
                    f"Required remote long-term {snapshot_kind!r} item {entry.item_id!r} is unavailable."
                )
            item_payloads.append(loaded)
        if snapshot_kind == "conflicts":
            assembled = {
                "schema": "twinr_memory_conflict_store",
                "version": 1,
                "conflicts": item_payloads,
            }
        else:
            assembled = {
                "schema": "twinr_memory_object_store" if snapshot_kind == "objects" else "twinr_memory_archive_store",
                "version": 1,
                "objects": item_payloads,
            }
        written_at = payload.get("written_at") if isinstance(payload, Mapping) else None
        if isinstance(written_at, str):
            assembled["written_at"] = written_at
        return assembled

    def load_item_payload(
        self,
        *,
        snapshot_kind: str,
        item_id: str,
        document_id: str | None = None,
        uri: str | None = None,
    ) -> dict[str, object] | None:
        """Load one remote item document exactly by document id or URI."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        definition = self._require_definition(snapshot_kind)
        normalized_item_id = self._normalize_text(item_id)
        if not normalized_item_id:
            return None
        resolved_uri = uri or self.item_uri(snapshot_kind=snapshot_kind, item_id=normalized_item_id)
        try:
            envelope = read_client.fetch_full_document(
                document_id=document_id,
                origin_uri=None if document_id else resolved_uri,
                include_content=True,
                max_content_chars=self._max_content_chars(),
            )
        except Exception as exc:
            if document_id:
                try:
                    envelope = read_client.fetch_full_document(
                        origin_uri=resolved_uri,
                        include_content=True,
                        max_content_chars=self._max_content_chars(),
                    )
                except Exception as fallback_exc:
                    raise LongTermRemoteUnavailableError(
                        f"Failed to read remote long-term {snapshot_kind!r} item {normalized_item_id!r}."
                    ) from fallback_exc
            else:
                raise LongTermRemoteUnavailableError(
                    f"Failed to read remote long-term {snapshot_kind!r} item {normalized_item_id!r}."
                ) from exc
        item_payload = self._extract_item_payload(
            definition=definition,
            item_id=normalized_item_id,
            payload=envelope,
        )
        if item_payload is None:
            return None
        return dict(item_payload)

    def load_item_payloads(
        self,
        *,
        snapshot_kind: str,
        item_ids: Iterable[str],
    ) -> tuple[dict[str, object], ...]:
        """Load a bounded set of current item payloads by Twinr item id."""

        entry_by_id = {
            entry.item_id: entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
        }
        loaded: list[dict[str, object]] = []
        for raw_item_id in item_ids:
            item_id = self._normalize_text(raw_item_id)
            if not item_id:
                continue
            entry = entry_by_id.get(item_id)
            if entry is None:
                continue
            payload = self.load_item_payload(
                snapshot_kind=snapshot_kind,
                item_id=item_id,
                document_id=entry.document_id,
                uri=entry.uri,
            )
            if payload is not None:
                loaded.append(payload)
        return tuple(loaded)

    def search_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        query_text: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Search one catalog through ChonkyDB retrieval restricted to current docs."""

        remote_state = self._require_remote_state()
        read_client = self._require_client(getattr(remote_state, "read_client", None), operation="read")
        entries = tuple(
            entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
            if eligible is None or eligible(entry)
        )
        if not entries:
            return ()
        by_item_id = {entry.item_id: entry for entry in entries}
        allowed_doc_ids = [
            entry.document_id
            for entry in entries
            if isinstance(entry.document_id, str) and entry.document_id
        ]
        if not allowed_doc_ids:
            return ()
        try:
            response = read_client.retrieve(
                ChonkyDBRetrieveRequest(
                    query_text=query_text,
                    result_limit=max(1, int(limit)),
                    include_content=False,
                    include_metadata=True,
                    allowed_doc_ids=tuple(allowed_doc_ids),
                )
            )
        except Exception as exc:
            raise LongTermRemoteUnavailableError(
                f"Failed to retrieve remote long-term {snapshot_kind!r} items."
            ) from exc

        selected: list[LongTermRemoteCatalogEntry] = []
        seen: set[str] = set()
        for hit in response.results:
            metadata = getattr(hit, "metadata", None)
            if not isinstance(metadata, Mapping):
                continue
            item_id = self._normalize_text(metadata.get("twinr_memory_item_id"))
            if not item_id or item_id in seen:
                continue
            entry = by_item_id.get(item_id)
            if entry is None:
                continue
            selected.append(entry)
            seen.add(item_id)
            if len(selected) >= max(1, int(limit)):
                break
        return tuple(selected)

    def top_catalog_entries(
        self,
        *,
        snapshot_kind: str,
        limit: int,
        eligible: Callable[[LongTermRemoteCatalogEntry], bool] | None = None,
        preserve_order: bool = False,
    ) -> tuple[LongTermRemoteCatalogEntry, ...]:
        """Return current catalog entries ordered either by recency or catalog order."""

        entries = [
            entry
            for entry in self.load_catalog_entries(snapshot_kind=snapshot_kind)
            if eligible is None or eligible(entry)
        ]
        if not preserve_order:
            entries.sort(key=lambda entry: (entry.updated_at(), entry.item_id), reverse=True)
        return tuple(entries[: max(1, int(limit))])

    def build_catalog_payload(
        self,
        *,
        snapshot_kind: str,
        item_payloads: Iterable[Mapping[str, object]],
        item_id_getter: Callable[[Mapping[str, object]], str | None],
        metadata_builder: Callable[[Mapping[str, object]], Mapping[str, object]],
        content_builder: Callable[[Mapping[str, object]], str],
    ) -> dict[str, object]:
        """Persist individual remote items and return the small current catalog."""

        remote_state = self._require_remote_state()
        write_client = self._require_client(getattr(remote_state, "write_client", None), operation="write")
        definition = self._require_definition(snapshot_kind)
        record_items: list[ChonkyDBRecordItem] = []
        staged: list[tuple[str, str, dict[str, object]]] = []
        for raw_payload in item_payloads:
            payload = dict(raw_payload)
            item_id = self._normalize_text(item_id_getter(payload))
            if not item_id:
                continue
            uri = self.item_uri(snapshot_kind=snapshot_kind, item_id=item_id)
            metadata = {
                "twinr_snapshot_kind": definition.snapshot_kind,
                "twinr_memory_item_id": item_id,
            }
            user_metadata = metadata_builder(payload)
            if isinstance(user_metadata, Mapping):
                for key, value in user_metadata.items():
                    if value is not None:
                        metadata[str(key)] = value
            envelope = {
                "schema": definition.item_schema,
                "version": _ITEM_VERSION,
                "snapshot_kind": definition.snapshot_kind,
                "item_id": item_id,
                definition.envelope_key: payload,
            }
            record_items.append(
                ChonkyDBRecordItem(
                    payload=envelope,
                    metadata=metadata,
                    content=content_builder(payload),
                    uri=uri,
                    enable_chunking=False,
                    include_insights_in_response=False,
                )
            )
            staged.append((item_id, uri, metadata))
        document_ids = self._store_record_items(write_client, record_items=record_items)
        catalog_entries = []
        for index, (item_id, uri, metadata) in enumerate(staged):
            catalog_entries.append(
                {
                    "item_id": item_id,
                    "document_id": document_ids[index] if index < len(document_ids) else None,
                    "uri": uri,
                    "metadata": metadata,
                }
            )
        return {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items": catalog_entries,
        }

    def item_uri(self, *, snapshot_kind: str, item_id: str) -> str:
        """Return the canonical URI for one fine-grained remote item."""

        remote_state = self._require_remote_state()
        definition = self._require_definition(snapshot_kind)
        namespace = quote(str(getattr(remote_state, "namespace", "") or "twinr_longterm_v1"), safe="")
        encoded_id = quote(item_id, safe="")
        return f"twinr://longterm/{namespace}/{definition.uri_segment}/{encoded_id}"

    def _store_record_items(
        self,
        write_client: object,
        *,
        record_items: list[ChonkyDBRecordItem],
    ) -> tuple[str | None, ...]:
        if not record_items:
            return ()
        batch_size = self._bulk_write_batch_size()
        document_ids: list[str | None] = []
        for start in range(0, len(record_items), batch_size):
            batch = record_items[start : start + batch_size]
            request = ChonkyDBBulkRecordRequest(
                items=tuple(batch),
                finalize_vector_segments=start + len(batch) >= len(record_items),
            )
            try:
                result = getattr(write_client, "store_records_bulk")(request)
            except Exception as exc:
                raise LongTermRemoteUnavailableError("Failed to persist fine-grained remote long-term memory items.") from exc
            document_ids.extend(self._extract_document_ids(result, expected=len(batch)))
        return tuple(document_ids)

    def _extract_document_ids(self, payload: object, *, expected: int) -> tuple[str | None, ...]:
        if not isinstance(payload, Mapping):
            return tuple(None for _ in range(expected))
        items = payload.get("items")
        if not isinstance(items, list):
            return tuple(None for _ in range(expected))
        document_ids: list[str | None] = []
        for item in items[:expected]:
            if not isinstance(item, Mapping):
                document_ids.append(None)
                continue
            document_ids.append(
                self._normalize_text(item.get("document_id"))
                or self._normalize_text(item.get("payload_id"))
                or self._normalize_text(item.get("chonky_id"))
            )
        while len(document_ids) < expected:
            document_ids.append(None)
        return tuple(document_ids)

    def _extract_item_payload(
        self,
        *,
        definition: _RemoteCollectionDefinition,
        item_id: str,
        payload: Mapping[str, object],
    ) -> Mapping[str, object] | None:
        for candidate in self._iter_record_candidates(payload):
            if candidate.get("schema") != definition.item_schema:
                continue
            if candidate.get("version") != _ITEM_VERSION:
                continue
            if candidate.get("snapshot_kind") != definition.snapshot_kind:
                continue
            if self._normalize_text(candidate.get("item_id")) != item_id:
                continue
            raw_item = candidate.get(definition.envelope_key)
            if isinstance(raw_item, Mapping):
                return raw_item
        return None

    def _iter_record_candidates(self, payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
        yield payload
        direct = payload.get("payload")
        if isinstance(direct, Mapping):
            yield direct
        nested = payload.get("record")
        if isinstance(nested, Mapping):
            nested_payload = nested.get("payload")
            if isinstance(nested_payload, Mapping):
                yield nested_payload
            nested_content = nested.get("content")
            if isinstance(nested_content, str):
                parsed = self._parse_json_mapping(nested_content)
                if parsed is not None:
                    yield parsed
        content = payload.get("content")
        if isinstance(content, str):
            parsed = self._parse_json_mapping(content)
            if parsed is not None:
                yield parsed

    def _parse_json_mapping(self, value: str) -> Mapping[str, object] | None:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, Mapping) else None

    def _load_catalog_payload(self, *, snapshot_kind: str) -> Mapping[str, object] | None:
        remote_state = self._require_remote_state()
        payload = remote_state.load_snapshot(snapshot_kind=snapshot_kind)
        return payload if isinstance(payload, Mapping) else None

    def _max_content_chars(self) -> int:
        remote_state = self._require_remote_state()
        value = getattr(getattr(remote_state, "config", None), "long_term_memory_remote_max_content_chars", None)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_MAX_ITEM_CONTENT_CHARS
        return max(1, resolved)

    def _bulk_write_batch_size(self) -> int:
        remote_state = self._require_remote_state()
        value = getattr(getattr(remote_state, "config", None), "long_term_memory_migration_batch_size", None)
        try:
            resolved = int(value)
        except (TypeError, ValueError):
            return _DEFAULT_BULK_BATCH_SIZE
        return max(1, resolved)

    def _definition(self, snapshot_kind: str) -> _RemoteCollectionDefinition | None:
        return _DEFINITIONS.get(self._normalize_text(snapshot_kind))

    def _require_definition(self, snapshot_kind: str) -> _RemoteCollectionDefinition:
        definition = self._definition(snapshot_kind)
        if definition is None:
            raise ValueError(f"Unsupported fine-grained remote snapshot kind {snapshot_kind!r}.")
        return definition

    def _require_remote_state(self) -> LongTermRemoteStateStore:
        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            raise LongTermRemoteUnavailableError("Required remote long-term memory state is not configured.")
        return remote_state

    def _require_client(self, client: object | None, *, operation: str) -> object:
        if client is not None:
            return client
        raise LongTermRemoteUnavailableError(
            f"Remote-primary long-term memory is enabled but ChonkyDB is not configured for {operation} operations."
        )

    @staticmethod
    def _normalize_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).strip()
        return normalized or None


__all__ = ["LongTermRemoteCatalogEntry", "LongTermRemoteCatalogStore"]
