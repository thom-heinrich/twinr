"""Bridge prompt callers onto the structured personality package."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.context_builder import PersonalityContextBuilder
from twinr.agent.personality.display_impulses import (
    AmbientDisplayImpulseCandidate,
    build_ambient_display_impulse_candidates,
)
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.intelligence.store import RemoteStateWorldIntelligenceStore
from twinr.agent.personality.models import PersonalitySnapshot
from twinr.agent.personality.positive_engagement import (
    PositiveEngagementTopicPolicy,
    build_positive_engagement_policies,
)
from twinr.agent.personality.steering import ConversationTurnSteeringCue, build_turn_steering_cues
from twinr.agent.personality.store import (
    PersonalitySnapshotStore,
    RemoteStatePersonalitySnapshotStore,
)
from twinr.memory.longterm.storage.remote_state import (
    LongTermRemoteStateStore,
    LongTermRemoteUnavailableError,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PersonalityContextService:
    """Load structured personality state and convert it into legacy sections."""

    builder: PersonalityContextBuilder = field(default_factory=PersonalityContextBuilder)
    store: PersonalitySnapshotStore = field(default_factory=RemoteStatePersonalitySnapshotStore)
    intelligence_store: RemoteStateWorldIntelligenceStore = field(default_factory=RemoteStateWorldIntelligenceStore)

    def load_snapshot(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> PersonalitySnapshot | None:
        """Load the optional structured personality snapshot for this turn."""

        try:
            return self.store.load_snapshot(config=config, remote_state=remote_state)
        except LongTermRemoteUnavailableError as exc:
            _LOGGER.warning(
                "Unable to load structured personality snapshot from remote state: %s",
                exc,
            )
            return None
        except ValueError:
            _LOGGER.exception("Ignoring malformed structured personality snapshot.")
            return None

    def build_static_sections(
        self,
        *,
        legacy_sections: tuple[tuple[str, str], ...],
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Merge legacy sections with any available structured personality state."""

        snapshot = self.load_snapshot(config=config, remote_state=remote_state)
        engagement_signals = self._load_engagement_signals(
            config=config,
            remote_state=remote_state,
        )
        plan = self.builder.build_prompt_plan(
            legacy_sections=legacy_sections,
            snapshot=snapshot,
            engagement_signals=engagement_signals,
        )
        return plan.as_sections()

    def build_supervisor_sections(
        self,
        *,
        legacy_sections: tuple[tuple[str, str], ...],
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Build a lean supervisor bundle without dynamic topic-contamination.

        The fast supervisor should keep stable character/style context, but it
        must not inherit volatile prompt layers such as `MINDSHARE`, `PLACE`,
        `WORLD`, or `REFLECTION`, because those layers can distort routing for
        noisy freshness-sensitive turns.
        """

        snapshot = self.load_snapshot(config=config, remote_state=remote_state)
        plan = self.builder.build_supervisor_prompt_plan(
            legacy_sections=legacy_sections,
            snapshot=snapshot,
        )
        return plan.as_sections()

    def load_turn_steering_cues(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        max_items: int = 3,
    ) -> tuple[ConversationTurnSteeringCue, ...]:
        """Load the bounded steering cues that may influence one turn.

        Args:
            config: Runtime configuration that points to remote personality and
                world-intelligence state.
            remote_state: Optional shared remote-state instance to reuse.
            max_items: Maximum number of cues to surface for the current turn.

        Returns:
            The current bounded steering cues derived from structured
            personality and world-intelligence state.
        """

        snapshot = self.load_snapshot(config=config, remote_state=remote_state)
        engagement_signals = self._load_engagement_signals(
            config=config,
            remote_state=remote_state,
        )
        return build_turn_steering_cues(
            snapshot,
            engagement_signals=engagement_signals,
            max_items=max_items,
        )

    def load_engagement_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldInterestSignal, ...]:
        """Load durable world-interest signals for display and turn shaping."""

        return self._load_engagement_signals(
            config=config,
            remote_state=remote_state,
        )

    def load_positive_engagement_policies(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        max_items: int = 3,
    ) -> tuple[PositiveEngagementTopicPolicy, ...]:
        """Load the current bounded positive-engagement topic actions."""

        snapshot = self.load_snapshot(config=config, remote_state=remote_state)
        engagement_signals = self.load_engagement_signals(
            config=config,
            remote_state=remote_state,
        )
        return build_positive_engagement_policies(
            snapshot,
            engagement_signals=engagement_signals,
            max_items=max_items,
        )

    def load_display_impulse_candidates(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
        local_now: datetime | None = None,
        max_items: int = 4,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Load the current bounded silent display-impulse candidates."""

        snapshot = self.load_snapshot(config=config, remote_state=remote_state)
        engagement_signals = self._load_engagement_signals(
            config=config,
            remote_state=remote_state,
        )
        return build_ambient_display_impulse_candidates(
            snapshot,
            engagement_signals=engagement_signals,
            local_now=local_now,
            max_items=max_items,
        )

    def _load_engagement_signals(
        self,
        *,
        config: TwinrConfig,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> tuple[WorldInterestSignal, ...]:
        """Load durable interest/engagement signals for mindshare surfacing."""

        try:
            state = self.intelligence_store.load_state(
                config=config,
                remote_state=remote_state,
            )
        except LongTermRemoteUnavailableError as exc:
            _LOGGER.warning(
                "Unable to load world-intelligence engagement state from remote state: %s",
                exc,
            )
            return ()
        except ValueError:
            _LOGGER.exception("Ignoring malformed world-intelligence state snapshot.")
            return ()
        return tuple(state.interest_signals)
