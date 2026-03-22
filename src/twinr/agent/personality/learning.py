"""Bridge personality signal extraction onto the background evolution loop.

This service is the runtime-facing entry point for implicit personality
learning. It keeps extraction and persistence separate:

- ``PersonalitySignalExtractor`` turns structured runtime artifacts into
  explicit interaction/place/world signal batches.
- ``BackgroundPersonalityEvolutionLoop`` persists those signals, applies
  policy-gated deltas, and commits the new promptable snapshot.

Callers should use this service from background/runtime integration points
instead of letting foreground prompt code mutate the personality directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.evolution import (
    BackgroundPersonalityEvolutionLoop,
    PersonalityEvolutionResult,
)
from twinr.agent.personality.intelligence import (
    WorldInterestSignalExtractor,
    WorldIntelligenceConfigRequest,
    WorldIntelligenceConfigResult,
    WorldIntelligenceRefreshResult,
    WorldInterestSignal,
    WorldIntelligenceService,
)
from twinr.agent.personality.signals import PersonalitySignalBatch, PersonalitySignalExtractor
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermConversationTurn,
)
from twinr.memory.longterm.storage.remote_state import LongTermRemoteStateStore


@dataclass(slots=True)
class PersonalityLearningService:
    """Record structured learning evidence and evolve the personality snapshot."""

    extractor: PersonalitySignalExtractor = field(default_factory=PersonalitySignalExtractor)
    world_interest_extractor: WorldInterestSignalExtractor = field(default_factory=WorldInterestSignalExtractor)
    background_loop: BackgroundPersonalityEvolutionLoop = field(
        default_factory=lambda: BackgroundPersonalityEvolutionLoop(config=TwinrConfig(project_root="."))
    )
    world_intelligence: WorldIntelligenceService | None = None
    _pending_world_interest_signals: list[WorldInterestSignal] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        remote_state: LongTermRemoteStateStore | None = None,
    ) -> "PersonalityLearningService":
        """Build a runtime learning service for one Twinr configuration."""

        return cls(
            extractor=PersonalitySignalExtractor(),
            world_interest_extractor=WorldInterestSignalExtractor(),
            background_loop=BackgroundPersonalityEvolutionLoop(
                config=config,
                remote_state=remote_state,
            ),
            world_intelligence=WorldIntelligenceService(
                config=config,
                remote_state=remote_state,
            ),
        )

    def record_conversation_consolidation(
        self,
        *,
        turn: LongTermConversationTurn,
        consolidation: LongTermConsolidationResultV1,
    ) -> PersonalityEvolutionResult:
        """Extract and commit signals from one consolidated conversation turn."""

        batch = self.extractor.extract_from_consolidation(
            turn=turn,
            consolidation=consolidation,
        )
        existing_interest_signals = ()
        if self.world_intelligence is not None:
            existing_interest_signals = self.world_intelligence.store.load_state(
                config=self.world_intelligence.config,
                remote_state=self.world_intelligence.remote_state,
            ).interest_signals
        world_interest_batch = self.world_interest_extractor.extract_from_personality_batch(
            turn_id=consolidation.turn_id,
            batch=batch,
            occurred_at=consolidation.occurred_at,
            existing_interest_signals=existing_interest_signals,
        )
        self._enqueue_batch(batch)
        self._enqueue_world_interest_signals(world_interest_batch.interest_signals)
        return self._flush_pending()

    def record_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> PersonalityEvolutionResult:
        """Extract and commit signals from tool call/result history."""

        batch = self.extractor.extract_from_tool_history(
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
        )
        world_interest_batch = self.world_interest_extractor.extract_from_tool_history(
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
        )
        self._enqueue_batch(batch)
        self._enqueue_world_interest_signals(world_interest_batch.interest_signals)
        return self._flush_pending()

    def enqueue_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> None:
        """Queue tool-history learning for the next background commit."""

        batch = self.extractor.extract_from_tool_history(
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
        )
        world_interest_batch = self.world_interest_extractor.extract_from_tool_history(
            tool_calls=tuple(tool_calls),
            tool_results=tuple(tool_results),
        )
        self._enqueue_batch(batch)
        self._enqueue_world_interest_signals(world_interest_batch.interest_signals)

    def flush_pending(self) -> PersonalityEvolutionResult | None:
        """Commit queued tool-history/world-interest signals when present."""

        if not self.background_loop.has_pending_items() and not self._pending_world_interest_signals:
            return None
        return self._flush_pending()

    def configure_world_intelligence(
        self,
        *,
        request: WorldIntelligenceConfigRequest,
        search_backend: object | None = None,
    ) -> WorldIntelligenceConfigResult:
        """Apply one explicit RSS/world-intelligence configuration change."""

        if self.world_intelligence is None:
            raise RuntimeError("world intelligence service is not configured")
        result = self.world_intelligence.configure(
            request=request,
            search_backend=search_backend,
        )
        self._commit_world_refresh(result.refresh)
        return result

    def maybe_refresh_world_intelligence(
        self,
        *,
        force: bool = False,
        search_backend: object | None = None,
    ) -> WorldIntelligenceRefreshResult | None:
        """Refresh due RSS subscriptions and commit their resulting context."""

        if self.world_intelligence is None:
            return None
        result = self.world_intelligence.maybe_refresh(
            force=force,
            search_backend=search_backend,
        )
        self._commit_world_refresh(result)
        return result

    def _record_world_interest_signals(self, signals) -> None:
        """Persist world-intelligence calibration evidence when available."""

        if self.world_intelligence is None or not signals:
            return
        self.world_intelligence.record_interest_signals(signals=tuple(signals))

    def _enqueue_batch(self, batch: PersonalitySignalBatch) -> None:
        """Append one extracted batch to the pending background queues."""

        for signal in batch.interaction_signals:
            self.background_loop.enqueue_interaction_signal(signal)
        for signal in batch.place_signals:
            self.background_loop.enqueue_place_signal(signal)
        for signal in batch.world_signals:
            self.background_loop.enqueue_world_signal(signal)
        for thread in batch.continuity_threads:
            self.background_loop.enqueue_continuity_thread(thread)

    def _enqueue_world_interest_signals(self, signals: Sequence[WorldInterestSignal]) -> None:
        """Queue world-interest calibration evidence for the next commit."""

        self._pending_world_interest_signals.extend(tuple(signals))

    def _flush_pending(self) -> PersonalityEvolutionResult:
        """Persist queued intelligence signals and commit pending personality updates."""

        if self._pending_world_interest_signals:
            self._record_world_interest_signals(tuple(self._pending_world_interest_signals))
            self._pending_world_interest_signals.clear()
        return self.background_loop.process_pending()

    def _commit(self, batch: PersonalitySignalBatch) -> PersonalityEvolutionResult:
        """Enqueue one extracted batch and commit it through the evolution loop."""

        self._enqueue_batch(batch)
        return self._flush_pending()

    def _commit_world_refresh(
        self,
        refresh_result: WorldIntelligenceRefreshResult | None,
    ) -> PersonalityEvolutionResult | None:
        """Commit refreshed world/continuity context through the background loop."""

        if refresh_result is None:
            return None
        batch = PersonalitySignalBatch(
            world_signals=tuple(refresh_result.world_signals),
            continuity_threads=tuple(refresh_result.continuity_threads),
        )
        if not batch.has_any():
            return None
        return self._commit(batch)
