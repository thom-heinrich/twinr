"""Local typing helpers for the refactored structured-store mixins."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import RLock
from typing import Any, TYPE_CHECKING

from twinr.memory.longterm.core.models import LongTermMemoryConflictV1, LongTermMemoryObjectV1
from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogStore

if TYPE_CHECKING:
    from .snapshots import StructuredStoreCurrentState


class StructuredStoreMixinBase:
    """Expose store-owned attributes through ``__getattr__`` for mixin typing.

    ``LongTermStructuredStoreBase`` owns these fields at runtime. The focused
    mixins only describe one behavior slice each, so this typing shim keeps
    mypy from treating them as attribute-less while preserving normal runtime
    ``AttributeError`` behavior if a field is genuinely missing.
    """

    _lock: RLock
    _remote_catalog: LongTermRemoteCatalogStore | None
    _recent_local_snapshot_payloads: dict[str, dict[str, object]]

    @property
    def objects_path(self) -> Path:
        raise AttributeError("objects_path")

    @property
    def conflicts_path(self) -> Path:
        raise AttributeError("conflicts_path")

    @property
    def archive_path(self) -> Path:
        raise AttributeError("archive_path")

    def _remote_catalog_enabled(self) -> bool:
        raise NotImplementedError

    def load_current_state_fine_grained_for_write(self) -> StructuredStoreCurrentState:
        raise NotImplementedError

    def load_objects_by_ids(self, memory_ids: Iterable[str]) -> tuple[LongTermMemoryObjectV1, ...]:
        raise NotImplementedError

    def load_objects_by_slot_keys(self, slot_keys: Iterable[str]) -> tuple[LongTermMemoryObjectV1, ...]:
        raise NotImplementedError

    def load_objects_by_event_ids(self, event_ids: Iterable[str]) -> tuple[LongTermMemoryObjectV1, ...]:
        raise NotImplementedError

    def load_objects_referencing_memory_ids(
        self,
        memory_ids: Iterable[str],
    ) -> tuple[LongTermMemoryObjectV1, ...]:
        raise NotImplementedError

    def load_conflicts_by_slot_keys(
        self,
        slot_keys: Iterable[str],
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        raise NotImplementedError

    def load_conflicts_for_memory_ids(
        self,
        memory_ids: Iterable[str],
    ) -> tuple[LongTermMemoryConflictV1, ...]:
        raise NotImplementedError

    def _conflict_doc_id(self, conflict: LongTermMemoryConflictV1) -> str:
        raise NotImplementedError

    def _persist_snapshot_payload(
        self,
        *,
        snapshot_kind: str,
        local_path: Path,
        payload: dict[str, object],
    ) -> None:
        raise NotImplementedError

    def _catalog_entry_text(self, entry: object, field: str) -> str:
        raise NotImplementedError

    def _remote_item_metadata(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        raise NotImplementedError

    def _remote_item_search_text(
        self,
        *,
        snapshot_kind: str,
        payload: Mapping[str, object],
    ) -> str:
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)
