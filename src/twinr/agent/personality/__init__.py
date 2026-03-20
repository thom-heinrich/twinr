"""Expose the layered agent-personality package surface.

This package owns the structured companion-identity models and prompt-layer
planning used to evolve Twinr beyond a single static personality markdown
file. Base-agent prompting remains the runtime caller; this package focuses on
typed state, prompt planning, and the future ChonkyDB/remote-state seam.
"""

from twinr.agent.personality.context_builder import PersonalityContextBuilder
from twinr.agent.personality.evolution import (
    BackgroundPersonalityEvolutionLoop,
    PersonalityEvolutionLoop,
    PersonalityEvolutionPolicy,
    PersonalityEvolutionResult,
)
from twinr.agent.personality.intelligence import (
    DEFAULT_WORLD_INTELLIGENCE_STATE_KIND,
    DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND,
    RemoteStateWorldIntelligenceStore,
    SituationalAwarenessThread,
    WorldFeedItem,
    WorldFeedSubscription,
    WorldIntelligenceConfigRequest,
    WorldIntelligenceConfigResult,
    WorldIntelligenceSignalBatch,
    WorldInterestSignal,
    WorldInterestSignalExtractor,
    WorldIntelligenceRefreshResult,
    WorldIntelligenceService,
    WorldIntelligenceState,
)
from twinr.agent.personality.learning import PersonalityLearningService
from twinr.agent.personality.models import (
    ConversationStyleProfile,
    DEFAULT_PERSONALITY_SNAPSHOT_KIND,
    ContinuityThread,
    HumorProfile,
    INTERACTION_SIGNAL_SNAPSHOT_KIND,
    InteractionSignal,
    PERSONALITY_DELTA_SNAPSHOT_KIND,
    PersonalityPromptLayer,
    PersonalityPromptPlan,
    PersonalitySnapshot,
    PersonalityTrait,
    PlaceFocus,
    PlaceSignal,
    PLACE_SIGNAL_SNAPSHOT_KIND,
    PersonalityDelta,
    ReflectionDelta,
    RelationshipSignal,
    WORLD_SIGNAL_SNAPSHOT_KIND,
    WorldSignal,
)
from twinr.agent.personality.service import PersonalityContextService
from twinr.agent.personality.signals import PersonalitySignalBatch, PersonalitySignalExtractor
from twinr.agent.personality.store import (
    PersonalityEvolutionStore,
    PersonalitySnapshotStore,
    RemoteStatePersonalityEvolutionStore,
    RemoteStatePersonalitySnapshotStore,
)

__all__ = [
    "BackgroundPersonalityEvolutionLoop",
    "ConversationStyleProfile",
    "DEFAULT_PERSONALITY_SNAPSHOT_KIND",
    "DEFAULT_WORLD_INTELLIGENCE_STATE_KIND",
    "DEFAULT_WORLD_INTELLIGENCE_SUBSCRIPTIONS_KIND",
    "ContinuityThread",
    "HumorProfile",
    "INTERACTION_SIGNAL_SNAPSHOT_KIND",
    "InteractionSignal",
    "PersonalityLearningService",
    "PERSONALITY_DELTA_SNAPSHOT_KIND",
    "PersonalityContextBuilder",
    "PersonalityContextService",
    "PersonalityDelta",
    "PersonalityEvolutionLoop",
    "PersonalityEvolutionPolicy",
    "PersonalityEvolutionResult",
    "PersonalityEvolutionStore",
    "PersonalityPromptLayer",
    "PersonalityPromptPlan",
    "PersonalitySignalBatch",
    "PersonalitySignalExtractor",
    "PersonalitySnapshot",
    "PersonalitySnapshotStore",
    "PersonalityTrait",
    "PlaceFocus",
    "PlaceSignal",
    "PLACE_SIGNAL_SNAPSHOT_KIND",
    "ReflectionDelta",
    "RemoteStateWorldIntelligenceStore",
    "RemoteStatePersonalityEvolutionStore",
    "RelationshipSignal",
    "RemoteStatePersonalitySnapshotStore",
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
    "WORLD_SIGNAL_SNAPSHOT_KIND",
    "WorldSignal",
]
