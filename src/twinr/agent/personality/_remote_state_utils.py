"""Shared remote-state resolution helpers for personality persistence seams."""

from __future__ import annotations

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


def resolve_remote_state(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
) -> LongTermRemoteStateStore | None:
    """Resolve the remote snapshot adapter for one load/save call."""

    if remote_state is not None:
        return remote_state
    resolved = LongTermRemoteStateStore.from_config(config)
    if not getattr(resolved, "enabled", False):
        return None
    return resolved
