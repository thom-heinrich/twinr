"""Remote-authoritative current-head persistence for midterm packets.

Midterm now stores its authoritative state as packet records plus a fixed
``.../midterm/catalog/current`` head. Fresh readers still need one bounded
compatibility bridge when that direct head lags cross-process visibility, so
this adapter also persists a small legacy snapshot head that contains only the
catalog payload. Reads stay current-head-first and only reuse that compatibility
head in a read-only way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timezone
import inspect
import logging

from twinr.memory.longterm.core.models import LongTermMidtermPacketV1
from twinr.memory.longterm.reasoning.turn_continuity import turn_continuity_recall_hints
from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


LOGGER = logging.getLogger(__name__)
_MIDTERM_SNAPSHOT_KIND = "midterm"


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _packet_updated_at_text(packet: LongTermMidtermPacketV1) -> str:
    updated_at = packet.updated_at
    if updated_at.tzinfo is None or updated_at.tzinfo.utcoffset(updated_at) is None:
        return updated_at.replace(tzinfo=timezone.utc).isoformat()
    return updated_at.astimezone(timezone.utc).isoformat()


def _packet_search_text(payload: Mapping[str, object]) -> str:
    parts: list[str] = [
        _normalize_text(payload.get("kind")),
        _normalize_text(payload.get("summary")),
        _normalize_text(payload.get("details")),
    ]
    query_hints = payload.get("query_hints")
    if isinstance(query_hints, (list, tuple)):
        parts.extend(_normalize_text(item) for item in query_hints if _normalize_text(item))
    attributes = payload.get("attributes")
    if isinstance(attributes, Mapping):
        for key, value in attributes.items():
            key_text = _normalize_text(key)
            if key_text:
                parts.append(key_text)
            if isinstance(value, str):
                value_text = _normalize_text(value)
                if value_text:
                    parts.append(value_text)
            elif isinstance(value, (list, tuple)):
                parts.extend(_normalize_text(item) for item in value if _normalize_text(item))
    parts.extend(
        turn_continuity_recall_hints(
            kind=payload.get("kind"),
            attributes=attributes if isinstance(attributes, Mapping) else None,
        )
    )
    return " ".join(part for part in parts if part)


def _packet_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kind": _normalize_text(payload.get("kind")),
        "summary": _normalize_text(payload.get("summary")),
        "updated_at": _normalize_text(payload.get("updated_at")),
        "created_at": _normalize_text(payload.get("created_at")),
        "valid_from": _normalize_text(payload.get("valid_from")),
        "valid_to": _normalize_text(payload.get("valid_to")),
        "selection_projection": dict(payload),
    }
    attributes = payload.get("attributes")
    if isinstance(attributes, Mapping):
        persistence_scope = _normalize_text(attributes.get("persistence_scope"))
        if persistence_scope:
            metadata["status"] = persistence_scope
    return {key: value for key, value in metadata.items() if value}


def _packet_from_payload(payload: Mapping[str, object]) -> LongTermMidtermPacketV1 | None:
    try:
        return LongTermMidtermPacketV1.from_payload(payload)
    except Exception:
        LOGGER.warning("Ignoring invalid remote midterm packet payload.", exc_info=True)
        return None


class LongTermRemoteMidtermState:
    """Persist and query remote midterm packets via fine-grained current-head records."""

    def __init__(self, remote_state: LongTermRemoteStateStore | None) -> None:
        self.remote_state = remote_state
        self._catalog = LongTermRemoteCatalogStore(remote_state)

    def enabled(self) -> bool:
        return bool(self.remote_state is not None and self.remote_state.enabled)

    def probe_current_head(self) -> dict[str, object] | None:
        if not self.enabled():
            return None
        direct_probe = self._catalog.probe_catalog_payload(snapshot_kind=_MIDTERM_SNAPSHOT_KIND)
        if isinstance(direct_probe, Mapping):
            return dict(direct_probe)
        return self._compatible_current_head_payload(use_probe=True)

    def current_head_payload(self) -> dict[str, object] | None:
        if not self.enabled():
            return None
        load_direct_head = getattr(self._catalog, "_load_catalog_head_payload", None)
        if callable(load_direct_head):
            direct_payload = load_direct_head(snapshot_kind=_MIDTERM_SNAPSHOT_KIND, metadata_only=False)
        else:
            direct_payload = self._catalog.load_catalog_payload(snapshot_kind=_MIDTERM_SNAPSHOT_KIND)
        if isinstance(direct_payload, Mapping):
            return dict(direct_payload)
        return self._compatible_current_head_payload(use_probe=False)

    def ensure_seeded(self, *, packets: Sequence[LongTermMidtermPacketV1]) -> bool:
        if not self.enabled():
            return False
        if isinstance(self.current_head_payload(), Mapping):
            return False
        self.save_packets(packets=packets)
        return True

    def save_packets(self, *, packets: Sequence[LongTermMidtermPacketV1]) -> dict[str, object] | None:
        if not self.enabled():
            return None
        packet_payloads = [packet.to_payload() for packet in packets]
        catalog_payload = self._catalog.build_catalog_payload(
            snapshot_kind=_MIDTERM_SNAPSHOT_KIND,
            item_payloads=packet_payloads,
            item_id_getter=lambda payload: payload.get("packet_id"),
            metadata_builder=_packet_metadata,
            content_builder=_packet_search_text,
            skip_async_document_id_wait=True,
        )
        if packets:
            catalog_payload["written_at"] = max(_packet_updated_at_text(packet) for packet in packets)
        self._catalog.persist_catalog_payload(
            snapshot_kind=_MIDTERM_SNAPSHOT_KIND,
            payload=catalog_payload,
            skip_async_document_id_wait=True,
        )
        self._persist_compatibility_current_head(payload=catalog_payload)
        return catalog_payload

    def _compatible_current_head_payload(self, *, use_probe: bool) -> dict[str, object] | None:
        """Load the small compatibility current head without promoting it.

        Fresh readers may observe the legacy snapshot URI before the fixed
        ``catalog/current`` document becomes visible. Reusing that tiny catalog
        payload keeps bootstrap and health checks read-only and prevents false
        reseeds when the direct head simply lags.
        """

        remote_state = self.remote_state
        if remote_state is None:
            return None
        payload: object | None = None
        probe_snapshot_load = getattr(remote_state, "probe_snapshot_load", None)
        if callable(probe_snapshot_load):
            probe_kwargs: dict[str, object] = {
                "snapshot_kind": _MIDTERM_SNAPSHOT_KIND,
                "prefer_cached_document_id": False,
                "prefer_metadata_only": True,
            }
            try:
                parameters = inspect.signature(probe_snapshot_load).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "fast_fail" in parameters:
                # Foreground current-head loads may spend the normal bounded
                # retry budget on the compatibility bridge. The single-attempt
                # readiness probe is only appropriate for explicit probe flows.
                probe_kwargs["fast_fail"] = use_probe
            probe = probe_snapshot_load(**probe_kwargs)
            payload = getattr(probe, "payload", None)
            if isinstance(payload, Mapping):
                payload_dict = dict(payload)
                if self._catalog.is_catalog_payload(snapshot_kind=_MIDTERM_SNAPSHOT_KIND, payload=payload_dict):
                    return payload_dict
            if use_probe:
                return None
            probe_status = _normalize_text(getattr(probe, "status", None)).lower()
            if probe_status == "not_found":
                return None
        if use_probe:
            return None
        else:
            load_snapshot = getattr(remote_state, "load_snapshot")
            load_kwargs: dict[str, object] = {"snapshot_kind": _MIDTERM_SNAPSHOT_KIND}
            try:
                parameters = inspect.signature(load_snapshot).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "prefer_cached_document_id" in parameters:
                load_kwargs["prefer_cached_document_id"] = False
            payload = load_snapshot(**load_kwargs)
        if not isinstance(payload, Mapping):
            return None
        payload_dict = dict(payload)
        if not self._catalog.is_catalog_payload(snapshot_kind=_MIDTERM_SNAPSHOT_KIND, payload=payload_dict):
            return None
        return payload_dict

    def _persist_compatibility_current_head(self, *, payload: Mapping[str, object]) -> None:
        """Mirror the small current-head payload onto the legacy snapshot URI.

        Midterm is no longer authoritative on the generic snapshot contract,
        but a bounded compatibility head is still required so fresh readers can
        prove the already-written state when the fixed current-head URI lags.
        """

        remote_state = self.remote_state
        if remote_state is None or not remote_state.enabled:
            return
        remote_state.save_snapshot(snapshot_kind=_MIDTERM_SNAPSHOT_KIND, payload=dict(payload))

    def load_packets(self) -> tuple[LongTermMidtermPacketV1, ...]:
        if not self.enabled():
            return ()
        entries = self._catalog.load_catalog_entries(snapshot_kind=_MIDTERM_SNAPSHOT_KIND)
        if not entries:
            return ()
        payloads = self._catalog.load_item_payloads(
            snapshot_kind=_MIDTERM_SNAPSHOT_KIND,
            item_ids=(entry.item_id for entry in entries),
        )
        packets: list[LongTermMidtermPacketV1] = []
        for payload in payloads:
            packet = _packet_from_payload(payload)
            if packet is not None:
                packets.append(packet)
        return tuple(packets)

    def current_packet_ids(self) -> tuple[str, ...]:
        if not self.enabled():
            return ()
        return tuple(entry.item_id for entry in self._catalog.load_catalog_entries(snapshot_kind=_MIDTERM_SNAPSHOT_KIND))

    def select_relevant_packets(
        self,
        *,
        query_text: str | None,
        limit: int,
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        if not self.enabled():
            return ()
        bounded_limit = max(0, int(limit))
        if bounded_limit == 0:
            return ()
        entries = self._catalog.load_catalog_entries(snapshot_kind=_MIDTERM_SNAPSHOT_KIND)
        if not entries:
            return ()
        clean_query = _normalize_text(query_text)
        if not clean_query:
            payloads = self._catalog.load_item_payloads(
                snapshot_kind=_MIDTERM_SNAPSHOT_KIND,
                item_ids=(entry.item_id for entry in entries[:bounded_limit]),
            )
        else:
            payloads = self._catalog.search_current_item_payloads(
                snapshot_kind=_MIDTERM_SNAPSHOT_KIND,
                query_text=clean_query,
                limit=bounded_limit,
                allow_catalog_fallback=True,
            ) or ()
        packets: list[LongTermMidtermPacketV1] = []
        for payload in payloads:
            packet = _packet_from_payload(payload)
            if packet is not None:
                packets.append(packet)
        return tuple(packets)
