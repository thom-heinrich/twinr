from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.fulltext import FullTextDocument, FullTextSelector
from twinr.memory.longterm.models import LongTermMidtermPacketV1, LongTermReflectionResultV1
from twinr.memory.longterm.remote_state import LongTermRemoteStateStore
from twinr.text_utils import retrieval_terms


_MIDTERM_STORE_SCHEMA = "twinr_memory_midterm_store"
_MIDTERM_STORE_VERSION = 1


def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(serialized, encoding="utf-8")
    temp_path.replace(path)


@dataclass(slots=True)
class LongTermMidtermStore:
    base_path: Path
    remote_state: LongTermRemoteStateStore | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    packet_type = LongTermMidtermPacketV1

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermMidtermStore":
        return cls(
            base_path=chonkydb_data_path(config),
            remote_state=LongTermRemoteStateStore.from_config(config),
        )

    @property
    def packets_path(self) -> Path:
        return self.base_path / "twinr_memory_midterm_v1.json"

    def load_packets(self) -> tuple[LongTermMidtermPacketV1, ...]:
        payload = self._load_payload()
        if payload is None:
            return ()
        items = payload.get("packets", [])
        if not isinstance(items, list):
            return ()
        return tuple(
            LongTermMidtermPacketV1.from_payload(item)
            for item in items
            if isinstance(item, dict)
        )

    def save_packets(self, *, packets: tuple[LongTermMidtermPacketV1, ...]) -> None:
        with self._lock:
            payload = {
                "schema": _MIDTERM_STORE_SCHEMA,
                "version": _MIDTERM_STORE_VERSION,
                "packets": [item.to_payload() for item in sorted(packets, key=lambda row: row.packet_id)],
            }
            if self.remote_state is not None and self.remote_state.enabled:
                self.remote_state.save_snapshot(snapshot_kind="midterm", payload=payload)
                return
            _write_json_atomic(self.packets_path, payload)

    def apply_reflection(self, result: LongTermReflectionResultV1) -> None:
        self.save_packets(packets=result.midterm_packets)

    def _load_payload(self) -> dict[str, object] | None:
        if self.remote_state is not None and self.remote_state.enabled:
            payload = self.remote_state.load_snapshot(snapshot_kind="midterm", local_path=self.packets_path)
            return payload
        if not self.packets_path.exists():
            return None
        loaded = json.loads(self.packets_path.read_text(encoding="utf-8"))
        return dict(loaded) if isinstance(loaded, dict) else None

    def select_relevant_packets(
        self,
        query_text: str | None,
        *,
        limit: int = 3,
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        packets = self.load_packets()
        if not packets:
            return ()
        clean_query = _normalize_text(query_text)
        if not clean_query:
            return packets[: max(1, limit)]
        selector = FullTextSelector(
            tuple(
                FullTextDocument(
                    doc_id=item.packet_id,
                    category="midterm",
                    content=self._packet_search_text(item),
                )
                for item in packets
            )
        )
        selected_ids = selector.search(clean_query, limit=max(1, limit), category="midterm")
        by_id = {item.packet_id: item for item in packets}
        selected = [by_id[packet_id] for packet_id in selected_ids if packet_id in by_id]
        return tuple(self._filter_query_relevant_packets(clean_query, selected=selected, limit=limit))

    def _packet_search_text(self, item: LongTermMidtermPacketV1) -> str:
        parts = [
            item.kind,
            item.summary,
            item.details or "",
            *item.query_hints,
        ]
        for key, value in (item.attributes or {}).items():
            parts.append(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple)):
                parts.extend(str(entry) for entry in value if isinstance(entry, str))
        return _normalize_text(" ".join(part for part in parts if part))

    def _filter_query_relevant_packets(
        self,
        query_text: str,
        *,
        selected: list[LongTermMidtermPacketV1],
        limit: int,
    ) -> tuple[LongTermMidtermPacketV1, ...]:
        query_terms = {term for term in retrieval_terms(query_text) if not term.isdigit()}
        if not query_terms:
            return tuple(selected[: max(1, limit)])
        filtered = [
            item
            for item in selected
            if query_terms.intersection(retrieval_terms(self._packet_search_text(item)))
        ]
        return tuple(filtered[: max(1, limit)])


__all__ = ["LongTermMidtermStore"]
