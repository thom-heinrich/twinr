"""Small base class that composes the focused remote-state mixins."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient

from .reads import LongTermRemoteStateReadMixin
from .shared import LongTermRemoteSnapshotProbe, _CachedSnapshotRead
from .state import LongTermRemoteStateSupportMixin
from .writes import LongTermRemoteStateWriteMixin


@dataclass(slots=True)
class LongTermRemoteStateStoreImpl(
    LongTermRemoteStateWriteMixin,
    LongTermRemoteStateReadMixin,
    LongTermRemoteStateSupportMixin,
):
    """Load and save remote snapshot state for long-term memory."""

    config: TwinrConfig
    read_client: ChonkyDBClient | None = None
    write_client: ChonkyDBClient | None = None
    namespace: str | None = None
    _state_lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)
    _circuit_open_until_monotonic: float = field(init=False, repr=False, default=0.0)
    _consecutive_failures: int = field(init=False, repr=False, default=0)
    _probe_cache_depth: int = field(init=False, repr=False, default=0)
    _probe_cache: dict[str, LongTermRemoteSnapshotProbe] = field(init=False, repr=False, default_factory=dict)
    _document_id_hints: dict[str, str] = field(init=False, repr=False, default_factory=dict)
    _snapshot_read_cache: dict[str, _CachedSnapshotRead] = field(init=False, repr=False, default_factory=dict)


__all__ = ["LongTermRemoteStateStoreImpl"]
