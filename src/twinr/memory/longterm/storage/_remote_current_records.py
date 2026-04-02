"""Remote current-head helpers for typed collections and one-item state records.

This module keeps small remote-authoritative state contracts off the generic
``remote_state.save_snapshot/load_snapshot`` blob path. Callers persist either:

- fine-grained typed collections via catalog heads plus individual item records
- a single typed current state by treating it as a one-item collection

The fixed current-head document remains authoritative; legacy snapshot heads may
still be written by the underlying catalog layer for compatibility, but callers
read/write through the current-head contract only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ._remote_catalog.shared import _CATALOG_VERSION
from .remote_catalog import LongTermRemoteCatalogStore
from .remote_state import LongTermRemoteStateStore

if TYPE_CHECKING:
    from .remote_catalog import LongTermRemoteCatalogEntry

_DEFAULT_SINGLE_RECORD_ITEM_ID = "current"


class LongTermRemoteCurrentRecordStore:
    """Persist typed remote state through fixed current heads plus item records."""

    def __init__(self, remote_state: LongTermRemoteStateStore | None) -> None:
        self.remote_state = remote_state
        self._catalog = LongTermRemoteCatalogStore(remote_state)

    def enabled(self) -> bool:
        """Return whether remote-authoritative current-head persistence is available."""

        return bool(self.remote_state is not None and self.remote_state.enabled)

    def probe_current_head(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Probe one current head without hydrating all item payloads."""

        if not self.enabled():
            return None
        return self._catalog.probe_catalog_payload(snapshot_kind=snapshot_kind)

    def probe_current_head_result(
        self,
        *,
        snapshot_kind: str,
    ) -> tuple[str, dict[str, object] | None]:
        """Probe one current head while preserving missing-vs-invalid status."""

        if not self.enabled():
            return "disabled", None
        return self._catalog.probe_catalog_payload_result(snapshot_kind=snapshot_kind)

    def load_current_head(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Load one current-head payload through the fixed-URI catalog contract."""

        if not self.enabled():
            return None
        return self._catalog.load_catalog_payload(snapshot_kind=snapshot_kind)

    def load_entries(self, *, snapshot_kind: str) -> tuple["LongTermRemoteCatalogEntry", ...]:
        """Load the current typed catalog entries for one collection."""

        if not self.enabled():
            return ()
        return self._catalog.load_catalog_entries(snapshot_kind=snapshot_kind)

    def current_item_ids(self, *, snapshot_kind: str) -> tuple[str, ...]:
        """Return the item ids referenced by the current head."""

        return tuple(entry.item_id for entry in self.load_entries(snapshot_kind=snapshot_kind))

    def load_collection_payloads(
        self,
        *,
        snapshot_kind: str,
        head_payload: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, object], ...]:
        """Hydrate every current item payload referenced by one known head.

        Callers may pass an already-probed current/legacy head payload to avoid
        reloading the remote head document and accidentally triggering the
        expensive legacy-head promotion path during watchdog/bootstrap reads.
        """

        if not self.enabled():
            return ()
        payload = dict(head_payload) if isinstance(head_payload, Mapping) else None
        entries = self._catalog.load_catalog_entries(
            snapshot_kind=snapshot_kind,
            payload=payload,
        )
        if not entries:
            return ()
        payloads = self._catalog._load_item_payloads_from_entries(
            snapshot_kind=snapshot_kind,
            entries=entries,
        )
        return tuple(dict(item_payload) for item_payload in payloads if isinstance(item_payload, Mapping))

    def probe_legacy_collection_head(
        self,
        *,
        snapshot_kind: str,
        prefer_metadata_only: bool = True,
    ) -> dict[str, object] | None:
        """Probe the legacy snapshot head without promoting it to `catalog/current`.

        Current-head callers use this read-only fallback when the fixed
        `.../catalog/current` document is missing. That keeps startup and
        watchdog health checks from blocking on a best-effort current-head
        promotion write before Twinr can prove the already-persisted legacy
        catalog payload is readable.
        """

        if not self.enabled():
            return None
        remote_state = self.remote_state
        if remote_state is None:
            return None
        probe_snapshot_load = getattr(remote_state, "probe_snapshot_load", None)
        if not callable(probe_snapshot_load):
            return None
        probe = probe_snapshot_load(
            snapshot_kind=snapshot_kind,
            prefer_cached_document_id=True,
            prefer_metadata_only=prefer_metadata_only,
        )
        payload = getattr(probe, "payload", None)
        if not isinstance(payload, Mapping):
            return None
        payload_dict = dict(payload)
        if not self._catalog.is_catalog_payload(snapshot_kind=snapshot_kind, payload=payload_dict):
            return None
        return payload_dict

    def empty_collection_head(self, *, snapshot_kind: str) -> dict[str, object] | None:
        """Return the canonical empty catalog head for one typed collection."""

        if not self.enabled():
            return None
        definition = self._catalog._require_definition(snapshot_kind)
        return {
            "schema": definition.catalog_schema,
            "version": _CATALOG_VERSION,
            "items_count": 0,
            "segments": [],
        }

    def save_empty_collection_head(
        self,
        *,
        snapshot_kind: str,
        head_fields: Mapping[str, object] | None = None,
        written_at: str | None = None,
        attest_readback: bool = True,
    ) -> dict[str, object] | None:
        """Publish one canonical empty current head without reading the old head.

        Empty prompt-context and prompt-memory writes do not need the previous
        catalog entries. Writing the fixed `.../catalog/current` head directly
        avoids unnecessary read-amplification when the old head is already known
        to be invalid or when the caller is intentionally deleting the final
        item from a collection.
        """

        if not self.enabled():
            return None
        payload = self.empty_collection_head(snapshot_kind=snapshot_kind)
        if not isinstance(payload, dict):
            return None
        if isinstance(written_at, str) and written_at.strip():
            payload["written_at"] = written_at.strip()
        if isinstance(head_fields, Mapping):
            for key, value in head_fields.items():
                if value is not None:
                    payload[str(key)] = value
        self._catalog.persist_catalog_payload(
            snapshot_kind=snapshot_kind,
            payload=payload,
            attest_readback=attest_readback,
        )
        return payload

    def load_single_payload(
        self,
        *,
        snapshot_kind: str,
        item_id: str = _DEFAULT_SINGLE_RECORD_ITEM_ID,
    ) -> dict[str, object] | None:
        """Load one typed current state stored as a one-item collection."""

        if not self.enabled():
            return None
        payloads = self._catalog.load_item_payloads(
            snapshot_kind=snapshot_kind,
            item_ids=(item_id,),
        )
        if payloads:
            return dict(payloads[0])
        entries = self._catalog.load_catalog_entries(snapshot_kind=snapshot_kind)
        if not entries:
            return None
        payloads = self._catalog.load_item_payloads(
            snapshot_kind=snapshot_kind,
            item_ids=(entries[0].item_id,),
        )
        if not payloads:
            return None
        return dict(payloads[0])

    def ensure_seeded_collection(
        self,
        *,
        snapshot_kind: str,
        item_payloads: Iterable[Mapping[str, object]],
        item_id_getter,
        metadata_builder,
        content_builder,
        head_fields: Mapping[str, object] | None = None,
        written_at: str | None = None,
    ) -> bool:
        """Create one missing current head plus item set."""

        if not self.enabled():
            return False
        if isinstance(self.load_current_head(snapshot_kind=snapshot_kind), Mapping):
            return False
        self.save_collection(
            snapshot_kind=snapshot_kind,
            item_payloads=item_payloads,
            item_id_getter=item_id_getter,
            metadata_builder=metadata_builder,
            content_builder=content_builder,
            head_fields=head_fields,
            written_at=written_at,
        )
        return True

    def ensure_seeded_single_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        metadata_builder,
        content_builder,
        item_id: str = _DEFAULT_SINGLE_RECORD_ITEM_ID,
        head_fields: Mapping[str, object] | None = None,
        written_at: str | None = None,
    ) -> bool:
        """Create one missing one-item current state."""

        if not self.enabled():
            return False
        if isinstance(self.load_current_head(snapshot_kind=snapshot_kind), Mapping):
            return False
        self.save_single_payload(
            snapshot_kind=snapshot_kind,
            payload=payload,
            metadata_builder=metadata_builder,
            content_builder=content_builder,
            item_id=item_id,
            head_fields=head_fields,
            written_at=written_at,
        )
        return True

    def save_collection(
        self,
        *,
        snapshot_kind: str,
        item_payloads: Iterable[Mapping[str, object]],
        item_id_getter,
        metadata_builder,
        content_builder,
        head_fields: Mapping[str, object] | None = None,
        written_at: str | None = None,
        replace_invalid_current_head: bool = False,
    ) -> dict[str, object] | None:
        """Persist a typed collection and publish its authoritative current head."""

        if not self.enabled():
            return None
        catalog_payload = self._catalog.build_catalog_payload(
            snapshot_kind=snapshot_kind,
            item_payloads=item_payloads,
            item_id_getter=item_id_getter,
            metadata_builder=metadata_builder,
            content_builder=content_builder,
            replace_invalid_current_head=replace_invalid_current_head,
        )
        if isinstance(written_at, str) and written_at.strip():
            catalog_payload["written_at"] = written_at.strip()
        if isinstance(head_fields, Mapping):
            for key, value in head_fields.items():
                if value is not None:
                    catalog_payload[str(key)] = value
        self._catalog.persist_catalog_payload(snapshot_kind=snapshot_kind, payload=catalog_payload)
        return catalog_payload

    def save_single_payload(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
        metadata_builder,
        content_builder,
        item_id: str = _DEFAULT_SINGLE_RECORD_ITEM_ID,
        head_fields: Mapping[str, object] | None = None,
        written_at: str | None = None,
        replace_invalid_current_head: bool = False,
    ) -> dict[str, object] | None:
        """Persist one typed current state as a one-item collection."""

        return self.save_collection(
            snapshot_kind=snapshot_kind,
            item_payloads=(payload,),
            item_id_getter=lambda _payload, item_id=item_id: item_id,
            metadata_builder=metadata_builder,
            content_builder=content_builder,
            head_fields=head_fields,
            written_at=written_at,
            replace_invalid_current_head=replace_invalid_current_head,
        )
