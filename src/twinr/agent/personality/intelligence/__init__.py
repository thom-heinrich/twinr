"""Expose RSS-backed world-intelligence helpers for the personality package.

This subpackage owns the persistent subscription and refresh logic that turns
curated RSS/Atom sources into calm place/world awareness for Twinr's evolving
personality. Feed discovery, refresh scheduling, and remote snapshot storage
stay separate from the main personality evolution loop so the agent can learn
from the world without turning every conversation into live news search.
"""

from twinr.agent.personality.intelligence.calibration import (
    WorldIntelligenceSignalBatch,
    WorldInterestSignalExtractor,
)
from twinr.agent.personality.intelligence.models import (
    DEFAULT_WORLD_INTELLIGENCE_STATE_KIND,
    DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND,
    SituationalAwarenessThread,
    WorldFeedItem,
    WorldFeedSubscription,
    WorldIntelligenceConfigRequest,
    WorldIntelligenceConfigResult,
    WorldInterestSignal,
    WorldIntelligenceRefreshResult,
    WorldIntelligenceState,
)
from twinr.agent.personality.intelligence.service import WorldIntelligenceService
from twinr.agent.personality.intelligence.store import (
    RemoteStateWorldIntelligenceStore,
    WorldIntelligenceStore,
)

__all__ = [
    "DEFAULT_WORLD_INTELLIGENCE_STATE_KIND",
    "DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND",
    "RemoteStateWorldIntelligenceStore",
    "SituationalAwarenessThread",
    "WorldFeedItem",
    "WorldFeedSubscription",
    "WorldIntelligenceConfigRequest",
    "WorldIntelligenceConfigResult",
    "WorldIntelligenceSignalBatch",
    "WorldInterestSignal",
    "WorldInterestSignalExtractor",
    "WorldIntelligenceRefreshResult",
    "WorldIntelligenceService",
    "WorldIntelligenceState",
    "WorldIntelligenceStore",
]
