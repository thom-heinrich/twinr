"""Shared remote-state resolution helpers for personality persistence seams."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.storage._remote_retry import clone_client_with_capped_timeout
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


def clone_remote_state_with_capped_read_timeout(
    *,
    config: TwinrConfig,
    remote_state: LongTermRemoteStateStore | None,
    timeout_s: float | None,
) -> LongTermRemoteStateStore | None:
    """Return one remote-state copy with a smaller read timeout when possible.

    Prompt-only personality/intelligence enrichments should fail open quickly
    when the backend spikes. The authoritative remote-primary path still uses
    the normal store timeouts elsewhere; this helper only narrows the optional
    read surface that decorates provider instructions.
    """

    if remote_state is None:
        return None
    capped_client = clone_client_with_capped_timeout(
        getattr(remote_state, "read_client", None),
        timeout_s=timeout_s,
    )
    if capped_client is getattr(remote_state, "read_client", None):
        return remote_state
    if is_dataclass(remote_state):
        try:
            return replace(remote_state, config=config, read_client=capped_client)
        except Exception:
            pass
    try:
        cloned_remote_state = copy.copy(remote_state)
    except Exception:
        return remote_state
    try:
        setattr(cloned_remote_state, "read_client", capped_client)
    except Exception:
        return remote_state
    return cloned_remote_state
