"""Orchestrate the full ambient companion candidate flow for the right lane.

The HDMI reserve lane should be planned from one coherent companion flow
instead of multiple partially overlapping helper paths. This module is that
integration layer. It blends:

- structured personality/mindshare candidates
- durable memory clarification and continuity prompts
- slower reflection-derived summaries and midterm packets
- long-horizon reserve-lane learning from prior visible card outcomes

The result is still just a bounded candidate pool. Scheduling, publishing, and
LLM copy rewriting remain in their dedicated modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.personality.display_impulses import (
    AmbientDisplayImpulseCandidate,
    build_ambient_display_impulse_candidates,
)
from twinr.agent.personality.intelligence.store import RemoteStateWorldIntelligenceStore
from twinr.agent.personality.service import PersonalityContextService
from twinr.memory.longterm.runtime.service import LongTermMemoryService

from .display_reserve_generation import DisplayReserveCopyGenerator
from .display_reserve_learning import (
    DisplayReserveLearningProfile,
    DisplayReserveLearningProfileBuilder,
)
from .display_reserve_memory import load_display_reserve_memory_candidates
from .display_reserve_reflection import load_display_reserve_reflection_candidates
from .display_reserve_world import load_display_reserve_world_candidates


def _compact_text(value: object | None) -> str:
    """Collapse arbitrary text into one trimmed single line."""

    return " ".join(str(value or "").split()).strip()


def _topic_key(value: object | None) -> str:
    """Return one stable dedupe/cooldown key."""

    return _compact_text(value).casefold()


def _candidate_sort_key(candidate: AmbientDisplayImpulseCandidate) -> tuple[float, str, str, str]:
    """Return one stable ranking key across all candidate families."""

    return (
        float(candidate.salience),
        _compact_text(candidate.attention_state).casefold(),
        _compact_text(candidate.action).casefold(),
        candidate.topic_key,
    )


@dataclass(frozen=True, slots=True)
class DisplayReserveCompanionFlowContext:
    """Collect the slower companion-flow inputs for one candidate pass."""

    snapshot: object | None
    learning_profile: DisplayReserveLearningProfile


@dataclass(slots=True)
class DisplayReserveCompanionFlow:
    """Blend ambient reserve candidates from personality, memory, and reflection."""

    personality_service: PersonalityContextService = field(default_factory=PersonalityContextService)
    world_store: RemoteStateWorldIntelligenceStore = field(default_factory=RemoteStateWorldIntelligenceStore)
    learning_builder_factory: type[DisplayReserveLearningProfileBuilder] = DisplayReserveLearningProfileBuilder
    copy_generator: DisplayReserveCopyGenerator = field(default_factory=DisplayReserveCopyGenerator)

    def load_candidates(
        self,
        config: TwinrConfig,
        *,
        local_now: datetime,
        max_items: int,
    ) -> tuple[AmbientDisplayImpulseCandidate, ...]:
        """Return one bounded companion-flow candidate set for the reserve lane."""

        limited_max = max(1, int(max_items))
        snapshot = self.personality_service.load_snapshot(config=config)
        engagement_signals = self.personality_service.load_engagement_signals(config=config)
        world_state = self.world_store.load_state(config=config)
        world_subscriptions = self.world_store.load_subscriptions(config=config)
        learning_profile = self.learning_builder_factory.from_config(config).build(now=local_now)
        memory_service = LongTermMemoryService.from_config(config)

        candidates: list[AmbientDisplayImpulseCandidate] = []
        candidates.extend(
            build_ambient_display_impulse_candidates(
                snapshot,
                engagement_signals=engagement_signals,
                local_now=local_now,
                max_items=max(limited_max, 6),
            )
        )
        candidates.extend(
            load_display_reserve_world_candidates(
                subscriptions=world_subscriptions,
                state=world_state,
                max_items=max(limited_max, 8),
            )
        )
        memory_candidates = load_display_reserve_memory_candidates(
            conflicts=memory_service.select_conflict_queue(
                query_text=None,
                limit=min(limited_max, 4),
            ),
            proactive_candidates=memory_service.plan_proactive_candidates(now=local_now).candidates,
            max_items=max(limited_max, 4),
        )
        candidates.extend(memory_candidates.as_tuple())
        candidates.extend(
            load_display_reserve_reflection_candidates(
                memory_service,
                config=config,
                local_now=local_now,
                max_items=max(limited_max, 4),
            )
        )

        adapted = tuple(
            self._apply_learning_profile(candidate, profile=learning_profile)
            for candidate in candidates
        )
        deduped = self._dedupe(adapted)
        ranked = sorted(deduped.values(), key=_candidate_sort_key, reverse=True)
        selected = tuple(ranked[:limited_max])
        return self.copy_generator.rewrite_candidates(
            config=config,
            snapshot=snapshot,
            candidates=selected,
            local_now=local_now,
        )

    def _apply_learning_profile(
        self,
        candidate: AmbientDisplayImpulseCandidate,
        *,
        profile: DisplayReserveLearningProfile,
    ) -> AmbientDisplayImpulseCandidate:
        """Bias one candidate by long-horizon reserve-lane learning."""

        adjustment = profile.candidate_adjustment(candidate)
        updated_context = dict(candidate.generation_context or {})
        updated_context["ambient_learning"] = profile.context_for_candidate(candidate)
        updated_reason = _compact_text(
            f"{candidate.reason}; ambient_learning={adjustment:+.2f}",
        )[:120]
        return replace(
            candidate,
            salience=max(0.0, min(1.25, float(candidate.salience) + adjustment)),
            reason=updated_reason or candidate.reason,
            generation_context=updated_context,
        )

    def _dedupe(
        self,
        candidates: Sequence[AmbientDisplayImpulseCandidate],
    ) -> dict[str, AmbientDisplayImpulseCandidate]:
        """Keep the strongest candidate for each stable topic key."""

        deduped: dict[str, AmbientDisplayImpulseCandidate] = {}
        for candidate in candidates:
            key = _topic_key(candidate.topic_key) or _topic_key(candidate.headline)
            if not key:
                continue
            current = deduped.get(key)
            if current is None or _candidate_sort_key(candidate) > _candidate_sort_key(current):
                deduped[key] = candidate
        return deduped


__all__ = ["DisplayReserveCompanionFlow", "DisplayReserveCompanionFlowContext"]
