# CHANGELOG: 2026-03-27
# BUG-1: Fold queued world-interest signals into novelty detection so back-to-back batches
# do not over-amplify the same topic before the next commit.
# BUG-2: Serialize enqueue/flush paths with an RLock so concurrent runtime/background callers
# cannot interleave pending mutations and silently duplicate or drop learning evidence.
# SEC-1: Bound pending in-memory learning with auto-flush thresholds and honor explicit
# untrusted world-refresh metadata to reduce practical Pi 4 DoS and memory-poisoning risk.
# IMP-1: Deduplicate world-interest signals with stable fingerprints before persistence.
# IMP-2: Accept optional refresh-produced interest signals and avoid repeated tool-history
# tuple materialization on the hot path.

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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import blake2b
from threading import RLock
from typing import Any

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import AgentToolCall, AgentToolResult
from twinr.agent.personality.ambient_feedback import AmbientImpulseFeedbackExtractor
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
    ambient_feedback_extractor: AmbientImpulseFeedbackExtractor | None = None
    world_interest_extractor: WorldInterestSignalExtractor = field(default_factory=WorldInterestSignalExtractor)
    background_loop: BackgroundPersonalityEvolutionLoop = field(
        default_factory=lambda: BackgroundPersonalityEvolutionLoop(config=TwinrConfig(project_root="."))
    )
    world_intelligence: WorldIntelligenceService | None = None

    # These thresholds keep pending state bounded on always-on, Pi-class deployments.
    max_pending_batch_items: int = 128
    max_pending_world_interest_signals: int = 256

    # Optional deployment hook for trust-aware world refresh gating.
    world_refresh_guard: Callable[[WorldIntelligenceRefreshResult], bool] | None = field(
        default=None,
        repr=False,
    )

    _pending_world_interest_signals: list[WorldInterestSignal] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _pending_world_interest_fingerprints: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _pending_background_items: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate memory-safety thresholds early."""

        if self.max_pending_batch_items <= 0:
            raise ValueError("max_pending_batch_items must be > 0")
        if self.max_pending_world_interest_signals <= 0:
            raise ValueError("max_pending_world_interest_signals must be > 0")

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
            ambient_feedback_extractor=AmbientImpulseFeedbackExtractor.from_config(config),
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

        with self._lock:
            batch = self.extractor.extract_from_consolidation(
                turn=turn,
                consolidation=consolidation,
            )
            if self.ambient_feedback_extractor is not None:
                batch = batch.merged(
                    self.ambient_feedback_extractor.extract_from_consolidation(
                        turn=turn,
                        consolidation=consolidation,
                        extracted_batch=batch,
                    )
                )

            world_interest_batch = self.world_interest_extractor.extract_from_personality_batch(
                turn_id=consolidation.turn_id,
                batch=batch,
                occurred_at=consolidation.occurred_at,
                existing_interest_signals=self._load_existing_world_interest_signals_locked(),
            )
            self._enqueue_batch_locked(batch)
            self._enqueue_world_interest_signals_locked(world_interest_batch.interest_signals)
            return self._flush_pending_locked()

    def record_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> PersonalityEvolutionResult:
        """Extract and commit signals from tool call/result history."""

        tool_call_tuple = tuple(tool_calls)
        tool_result_tuple = tuple(tool_results)

        with self._lock:
            batch = self.extractor.extract_from_tool_history(
                tool_calls=tool_call_tuple,
                tool_results=tool_result_tuple,
            )
            world_interest_batch = self.world_interest_extractor.extract_from_tool_history(
                tool_calls=tool_call_tuple,
                tool_results=tool_result_tuple,
            )
            self._enqueue_batch_locked(batch)
            self._enqueue_world_interest_signals_locked(world_interest_batch.interest_signals)
            return self._flush_pending_locked()

    def enqueue_tool_history(
        self,
        *,
        tool_calls: Sequence[AgentToolCall],
        tool_results: Sequence[AgentToolResult],
    ) -> None:
        """Queue tool-history learning for the next background commit."""

        tool_call_tuple = tuple(tool_calls)
        tool_result_tuple = tuple(tool_results)

        with self._lock:
            batch = self.extractor.extract_from_tool_history(
                tool_calls=tool_call_tuple,
                tool_results=tool_result_tuple,
            )
            world_interest_batch = self.world_interest_extractor.extract_from_tool_history(
                tool_calls=tool_call_tuple,
                tool_results=tool_result_tuple,
            )
            self._enqueue_batch_locked(batch)
            self._enqueue_world_interest_signals_locked(world_interest_batch.interest_signals)

            # BREAKING: enqueue_tool_history() can now synchronously flush once the pending
            # in-memory learning budget is exceeded. This bounds RAM usage on Pi-class devices.
            if self._should_auto_flush_locked():
                self._flush_pending_locked()

    def flush_pending(self) -> PersonalityEvolutionResult | None:
        """Commit queued tool-history/world-interest signals when present."""

        with self._lock:
            if not self._has_any_pending_locked():
                return None
            return self._flush_pending_locked()

    def configure_world_intelligence(
        self,
        *,
        request: WorldIntelligenceConfigRequest,
        search_backend: object | None = None,
    ) -> WorldIntelligenceConfigResult:
        """Apply one explicit RSS/world-intelligence configuration change."""

        with self._lock:
            if self.world_intelligence is None:
                raise RuntimeError("world intelligence service is not configured")
            result = self.world_intelligence.configure(
                request=request,
                search_backend=search_backend,
            )
            self._commit_world_refresh_locked(result.refresh)
            return result

    def maybe_refresh_world_intelligence(
        self,
        *,
        force: bool = False,
        search_backend: object | None = None,
    ) -> WorldIntelligenceRefreshResult | None:
        """Refresh due RSS subscriptions and commit their resulting context."""

        with self._lock:
            if self.world_intelligence is None:
                return None
            result = self.world_intelligence.maybe_refresh(
                force=force,
                search_backend=search_backend,
            )
            self._commit_world_refresh_locked(result)
            return result

    def _record_world_interest_signals(
        self,
        signals: Sequence[WorldInterestSignal],
    ) -> None:
        """Persist world-intelligence calibration evidence when available."""

        if self.world_intelligence is None or not signals:
            return
        self.world_intelligence.record_interest_signals(signals=tuple(signals))

    def _enqueue_batch_locked(self, batch: PersonalitySignalBatch) -> None:
        """Append one extracted batch to the pending background queues."""

        if not batch.has_any():
            return

        for signal in batch.interaction_signals:
            self.background_loop.enqueue_interaction_signal(signal)
        for signal in batch.place_signals:
            self.background_loop.enqueue_place_signal(signal)
        for signal in batch.world_signals:
            self.background_loop.enqueue_world_signal(signal)
        for thread in batch.continuity_threads:
            self.background_loop.enqueue_continuity_thread(thread)

        self._pending_background_items += (
            len(batch.interaction_signals)
            + len(batch.place_signals)
            + len(batch.world_signals)
            + len(batch.continuity_threads)
        )

    def _enqueue_world_interest_signals_locked(
        self,
        signals: Sequence[WorldInterestSignal],
    ) -> None:
        """Queue world-interest calibration evidence for the next commit."""

        for signal in signals:
            fingerprint = self._signal_fingerprint(signal)
            if fingerprint in self._pending_world_interest_fingerprints:
                continue
            self._pending_world_interest_signals.append(signal)
            self._pending_world_interest_fingerprints.add(fingerprint)

    def _flush_pending_locked(self) -> PersonalityEvolutionResult:
        """Persist queued intelligence signals and commit pending personality updates."""

        self._persist_pending_world_interest_signals_locked()
        result = self.background_loop.process_pending()
        self._pending_background_items = 0
        return result

    def _persist_pending_world_interest_signals_locked(self) -> None:
        """Persist queued world-interest evidence exactly once per successful write."""

        if not self._pending_world_interest_signals:
            return

        self._record_world_interest_signals(tuple(self._pending_world_interest_signals))
        self._pending_world_interest_signals.clear()
        self._pending_world_interest_fingerprints.clear()

    def _commit_locked(self, batch: PersonalitySignalBatch) -> PersonalityEvolutionResult:
        """Enqueue one extracted batch and commit it through the evolution loop."""

        self._enqueue_batch_locked(batch)
        return self._flush_pending_locked()

    def _commit_world_refresh_locked(
        self,
        refresh_result: WorldIntelligenceRefreshResult | None,
    ) -> PersonalityEvolutionResult | None:
        """Commit refreshed world/continuity context through the background loop."""

        if refresh_result is None:
            return None

        # BREAKING: refresh payloads that explicitly declare themselves untrusted
        # are now quarantined and never committed into long-term personality state.
        if not self._is_world_refresh_allowed_locked(refresh_result):
            return None

        refresh_interest_signals = tuple(getattr(refresh_result, "interest_signals", ()))
        if refresh_interest_signals:
            self._enqueue_world_interest_signals_locked(refresh_interest_signals)

        batch = PersonalitySignalBatch(
            world_signals=tuple(refresh_result.world_signals),
            continuity_threads=tuple(refresh_result.continuity_threads),
        )
        if not batch.has_any():
            self._persist_pending_world_interest_signals_locked()
            return None

        return self._commit_locked(batch)

    def _has_any_pending_locked(self) -> bool:
        """Return whether either the evolution loop or interest queue has pending work."""

        return (
            self._pending_background_items > 0
            or self.background_loop.has_pending_items()
            or bool(self._pending_world_interest_signals)
        )

    def _should_auto_flush_locked(self) -> bool:
        """Bound pending in-memory learning to protect Pi-class devices."""

        return (
            self._pending_background_items >= self.max_pending_batch_items
            or len(self._pending_world_interest_signals) >= self.max_pending_world_interest_signals
        )

    def _load_existing_world_interest_signals_locked(self) -> tuple[WorldInterestSignal, ...]:
        """Load persisted interest state plus any local-yet-unflushed evidence."""

        persisted: tuple[WorldInterestSignal, ...] = ()
        if self.world_intelligence is not None:
            persisted = tuple(
                self.world_intelligence.store.load_state(
                    config=self.world_intelligence.config,
                    remote_state=self.world_intelligence.remote_state,
                ).interest_signals
            )

        if not self._pending_world_interest_signals:
            return persisted

        return self._dedupe_world_interest_signals((*persisted, *self._pending_world_interest_signals))

    def _dedupe_world_interest_signals(
        self,
        signals: Sequence[WorldInterestSignal],
    ) -> tuple[WorldInterestSignal, ...]:
        """Return signals in stable order without duplicate semantic payloads."""

        deduped: list[WorldInterestSignal] = []
        seen: set[str] = set()

        for signal in signals:
            fingerprint = self._signal_fingerprint(signal)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(signal)

        return tuple(deduped)

    def _signal_fingerprint(self, signal: object) -> str:
        """Generate a stable, process-local fingerprint for deduplication."""

        frozen = self._freeze_for_fingerprint(signal)
        payload = repr(frozen).encode("utf-8", "backslashreplace")
        return blake2b(payload, digest_size=16).hexdigest()

    def _freeze_for_fingerprint(self, value: Any) -> Any:
        """Convert nested signal objects into a deterministic representation."""

        if is_dataclass(value) and not isinstance(value, type):
            return tuple(
                (dataclass_field.name, self._freeze_for_fingerprint(getattr(value, dataclass_field.name)))
                for dataclass_field in fields(value)
            )

        if isinstance(value, dict):
            return tuple(
                (repr(key), self._freeze_for_fingerprint(item))
                for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
            )

        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_for_fingerprint(item) for item in value)

        if isinstance(value, set):
            return tuple(sorted((self._freeze_for_fingerprint(item) for item in value), key=repr))

        slot_names = getattr(type(value), "__slots__", ())
        if slot_names:
            if isinstance(slot_names, str):
                slot_names = (slot_names,)
            return tuple(
                (slot_name, self._freeze_for_fingerprint(getattr(value, slot_name)))
                for slot_name in slot_names
                if hasattr(value, slot_name)
            )

        if hasattr(value, "__dict__"):
            return self._freeze_for_fingerprint(vars(value))

        return value

    def _is_world_refresh_allowed_locked(
        self,
        refresh_result: WorldIntelligenceRefreshResult,
    ) -> bool:
        """Respect explicit trust metadata when the upstream refresh exposes it."""

        if self.world_refresh_guard is not None:
            return self.world_refresh_guard(refresh_result)

        for attribute_name in ("is_trusted", "trusted", "source_trusted"):
            attribute_value = getattr(refresh_result, attribute_name, None)
            if attribute_value is False:
                return False

        provenance = getattr(refresh_result, "provenance", None)
        if provenance is not None and getattr(provenance, "trusted", None) is False:
            return False

        trust_score = getattr(refresh_result, "trust_score", None)
        minimum_trust_score = getattr(refresh_result, "minimum_trust_score", None)
        if (
            trust_score is not None
            and minimum_trust_score is not None
            and trust_score < minimum_trust_score
        ):
            return False

        return True