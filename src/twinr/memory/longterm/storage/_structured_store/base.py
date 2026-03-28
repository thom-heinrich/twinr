"""Small base class that composes the focused structured-store mixins."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb.client import chonkydb_data_path
from twinr.memory.longterm.storage.remote_catalog import LongTermRemoteCatalogStore
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore

from .mutations import StructuredStoreMutationMixin
from .query import StructuredStoreQueryMixin
from .remote_select import StructuredStoreRemoteSelectionMixin
from .retrieval import StructuredStoreRetrievalMixin
from .snapshots import StructuredStoreSnapshotMixin


@dataclass(slots=True)
class LongTermStructuredStoreBase(
    StructuredStoreSnapshotMixin,
    StructuredStoreRemoteSelectionMixin,
    StructuredStoreMutationMixin,
    StructuredStoreRetrievalMixin,
    StructuredStoreQueryMixin,
):
    """Read, mutate, and mirror durable long-term memory snapshots."""

    base_path: Path
    remote_state: LongTermRemoteStateStore | None = None
    _lock: RLock = field(default_factory=RLock, repr=False)
    _remote_catalog: LongTermRemoteCatalogStore | None = field(init=False, repr=False, default=None)
    _recent_local_snapshot_payloads: dict[str, dict[str, object]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the configured base path once during construction."""

        self.base_path = Path(self.base_path).expanduser().resolve(strict=False)
        self._remote_catalog = LongTermRemoteCatalogStore(self.remote_state)

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LongTermStructuredStoreBase":
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
