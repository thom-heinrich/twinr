"""Compatibility wrapper for the ambient companion reserve-candidate flow.

Historically the right-lane candidate loader lived entirely in this module.
The ambient companion flow is now split into dedicated modules for memory
hooks, reflection sources, long-horizon learning, and orchestration. This file
keeps the public loader stable while delegating the actual work to that
modular flow layer.
"""

from __future__ import annotations

from datetime import datetime

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import AmbientDisplayImpulseCandidate

from .display_reserve_flow import DisplayReserveCompanionFlow


def load_display_reserve_candidates(
    config: TwinrConfig,
    *,
    local_now: datetime,
    max_items: int,
) -> tuple[AmbientDisplayImpulseCandidate, ...]:
    """Load the full right-lane candidate set for the current local moment."""

    return DisplayReserveCompanionFlow().load_candidates(
        config,
        local_now=local_now,
        max_items=max_items,
    )


__all__ = ["load_display_reserve_candidates"]
