"""Shared remote-state resolution helpers for personality persistence seams."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
import math
from typing import cast

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.chonkydb import ChonkyDBClient
from twinr.memory.longterm.storage._remote_retry import clone_client_with_capped_timeout
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


_READ_TIMEOUT_CAP_ATTR = "_twinr_timeout_cap_s"


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
    """Return one remote-state copy with a smaller read timeout when possible."""

    if remote_state is None:
        return None
    timeout_cap_s: float | None = None
    if timeout_s is not None:
        try:
            parsed_timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            parsed_timeout_s = float("nan")
        if math.isfinite(parsed_timeout_s):
            timeout_cap_s = max(0.1, parsed_timeout_s)
    if timeout_cap_s is None:
        return remote_state
    capped_client = clone_client_with_capped_timeout(
        getattr(remote_state, "read_client", None),
        timeout_s=timeout_cap_s,
    )
    if capped_client is getattr(remote_state, "read_client", None):
        return remote_state
    if timeout_cap_s is not None:
        try:
            setattr(capped_client, _READ_TIMEOUT_CAP_ATTR, timeout_cap_s)
        except Exception:
            pass
    if is_dataclass(remote_state):
        return replace(
            remote_state,
            config=config,
            read_client=cast(ChonkyDBClient | None, capped_client),
        )
    cloned_remote_state = copy.copy(remote_state)
    setattr(cloned_remote_state, "read_client", capped_client)
    return cloned_remote_state
