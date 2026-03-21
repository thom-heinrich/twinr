"""Bridge prompt callers onto the structured personality package."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.context_builder import PersonalityContextBuilder
from twinr.agent.personality.intelligence.models import WorldInterestSignal
from twinr.agent.personality.intelligence.store import RemoteStateWorldIntelligenceStore
from twinr.agent.personality.models import PersonalitySnapshot
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
