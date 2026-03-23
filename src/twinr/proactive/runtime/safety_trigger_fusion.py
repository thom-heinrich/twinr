"""Bridge short-window event fusion into the active runtime safety path.

This module keeps ``service.py`` orchestration-only. It composes the existing
``SocialTriggerEngine`` with the newer ``event_fusion`` tracker and prefers
fused safety claims when they are available, while preserving the legacy
engine as the fallback path for non-safety triggers and visibility-loss cases
that the current V1 fusion tracker does not yet model directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.event_fusion import FusedEventClaim, MultimodalEventFusionTracker
from twinr.proactive.social.engine import (
    SocialObservation,
    SocialTriggerDecision,
    SocialTriggerEngine,
    SocialTriggerEvaluation,
    SocialTriggerPriority,
    TriggerScoreEvidence,
)


_FUSED_POSSIBLE_FALL_THRESHOLD = 0.72
_FUSED_FLOOR_STILLNESS_THRESHOLD = 0.74
_FUSED_DISTRESS_THRESHOLD = 0.70


@dataclass(frozen=True, slots=True)
class _FusedTriggerSpec:
    """Describe how one fused claim maps onto a runtime trigger contract."""

    trigger_id: str
    priority: SocialTriggerPriority
    threshold: float
    prompt: str
    reason: str


class SafetyTriggerFusionBridge:
    """Select runtime trigger decisions from engine output plus fused claims."""

    def __init__(
        self,
        *,
        config: TwinrConfig,
        engine: SocialTriggerEngine | object,
        tracker: MultimodalEventFusionTracker | None = None,
    ) -> None:
        """Initialize one bridge from the runtime config and trigger engine."""

        self.config = config
        self.engine = engine
        self.tracker = tracker or MultimodalEventFusionTracker()
        self._user_name = self._coerce_display_name(
            getattr(engine, "user_name", None) or getattr(config, "user_display_name", None)
        )
        self._last_fused_claims: tuple[FusedEventClaim, ...] = ()
        self._last_fused_evaluations: tuple[SocialTriggerEvaluation, ...] = ()
        self._last_fused_claim_by_trigger_id: dict[str, FusedEventClaim] = {}
        self._last_selected_source: str | None = None
        self._last_selected_claim: FusedEventClaim | None = None

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        engine: SocialTriggerEngine | object,
    ) -> "SafetyTriggerFusionBridge":
        """Build one bridge from the canonical runtime config."""

        return cls(config=config, engine=engine)

    @property
    def best_evaluation(self) -> SocialTriggerEvaluation | None:
        """Return the strongest latest candidate across engine and fusion."""

        engine_best = getattr(self.engine, "best_evaluation", None)
        fused_best = self.best_fused_evaluation
        candidates = tuple(item for item in (engine_best, fused_best) if item is not None)
        if not candidates:
            return None
        passed = tuple(item for item in candidates if item.passed)
        if passed:
            return max(passed, key=lambda item: (int(item.priority), item.score))
        return max(candidates, key=lambda item: (item.score, int(item.priority)))

    @property
    def best_fused_evaluation(self) -> SocialTriggerEvaluation | None:
        """Return the strongest latest fused evaluation, including near misses."""

        if not self._last_fused_evaluations:
            return None
        passed = tuple(item for item in self._last_fused_evaluations if item.passed)
        if passed:
            return max(passed, key=lambda item: (int(item.priority), item.score))
        return max(self._last_fused_evaluations, key=lambda item: (item.score, int(item.priority)))

    @property
    def last_fused_claims(self) -> tuple[FusedEventClaim, ...]:
        """Return the raw fused claims from the latest observation tick."""

        return self._last_fused_claims

    @property
    def last_fused_evaluations(self) -> tuple[SocialTriggerEvaluation, ...]:
        """Return the mapped fused safety evaluations from the latest tick."""

        return self._last_fused_evaluations

    @property
    def last_selected_source(self) -> str | None:
        """Return the latest decision source: ``social_engine`` or ``event_fusion``."""

        return self._last_selected_source

    @property
    def last_selected_claim(self) -> FusedEventClaim | None:
        """Return the latest fused claim that won runtime selection."""

        return self._last_selected_claim

    def observe(
        self,
        observation: SocialObservation,
        *,
        room_busy_or_overlapping: bool = False,
    ) -> SocialTriggerDecision | None:
        """Observe one tick and choose the runtime trigger decision."""

        engine_decision = self._observe_engine(observation)
        self._last_fused_claims = self.tracker.observe(
            observation,
            room_busy_or_overlapping=room_busy_or_overlapping,
        )
        (
            self._last_fused_evaluations,
            self._last_fused_claim_by_trigger_id,
        ) = self._evaluate_fused_claims(
            observed_at=observation.observed_at,
            claims=self._last_fused_claims,
        )
        fused_evaluation = self.best_fused_evaluation
        fused_decision = None if fused_evaluation is None or not fused_evaluation.passed else self._decision_from_evaluation(
            fused_evaluation
        )
        selected = self._select_runtime_decision(
            engine_decision=engine_decision,
            fused_decision=fused_decision,
        )
        if selected is not None and selected is fused_decision:
            note_dispatched = getattr(self.engine, "note_trigger_dispatched", None)
            if callable(note_dispatched):
                note_dispatched(selected.trigger_id, observed_at=selected.observed_at)
        return selected

    def _observe_engine(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        """Delegate one observation to the legacy social trigger engine."""

        observe = getattr(self.engine, "observe", None)
        if not callable(observe):
            return None
        return observe(observation)

    def _select_runtime_decision(
        self,
        *,
        engine_decision: SocialTriggerDecision | None,
        fused_decision: SocialTriggerDecision | None,
    ) -> SocialTriggerDecision | None:
        """Prefer the strongest available decision across engine and fusion."""

        if fused_decision is None and engine_decision is None:
            self._last_selected_source = None
            self._last_selected_claim = None
            return None
        if fused_decision is None:
            self._last_selected_source = "social_engine"
            self._last_selected_claim = None
            return engine_decision
        if engine_decision is None:
            self._last_selected_source = "event_fusion"
            self._last_selected_claim = self._last_fused_claim_by_trigger_id.get(fused_decision.trigger_id)
            return fused_decision

        fused_key = (int(fused_decision.priority), fused_decision.score)
        engine_key = (int(engine_decision.priority), engine_decision.score)
        if fused_key >= engine_key:
            self._last_selected_source = "event_fusion"
            self._last_selected_claim = self._last_fused_claim_by_trigger_id.get(fused_decision.trigger_id)
            return fused_decision
        self._last_selected_source = "social_engine"
        self._last_selected_claim = None
        return engine_decision

    def _evaluate_fused_claims(
        self,
        *,
        observed_at: float,
        claims: tuple[FusedEventClaim, ...],
    ) -> tuple[tuple[SocialTriggerEvaluation, ...], dict[str, FusedEventClaim]]:
        """Map latest fused claims onto bounded runtime safety evaluations."""

        evaluations_by_trigger: dict[str, SocialTriggerEvaluation] = {}
        claims_by_trigger: dict[str, FusedEventClaim] = {}
        engine_evaluations = {
            item.trigger_id: item
            for item in tuple(getattr(self.engine, "last_evaluations", ()) or ())
            if isinstance(item, SocialTriggerEvaluation)
        }
        for claim in claims:
            spec = self._spec_for_claim(claim)
            if spec is None:
                continue
            blocked_reason = None
            if not claim.active:
                blocked_reason = "claim_inactive"
            elif not claim.delivery_allowed:
                blocked_reason = "claim_blocked_" + "_".join(claim.blocked_by or ("policy",))
            engine_evaluation = engine_evaluations.get(spec.trigger_id)
            if blocked_reason is None and engine_evaluation is not None and engine_evaluation.blocked_reason == "cooldown_active":
                blocked_reason = "cooldown_active"
            evidence = self._evidence_from_claim(claim)
            evaluation = SocialTriggerEvaluation(
                trigger_id=spec.trigger_id,
                prompt=spec.prompt,
                reason=spec.reason,
                observed_at=observed_at,
                priority=spec.priority,
                score=claim.confidence,
                threshold=spec.threshold,
                evidence=evidence,
                passed=(blocked_reason is None and claim.confidence >= spec.threshold),
                blocked_reason=blocked_reason,
            )
            previous = evaluations_by_trigger.get(spec.trigger_id)
            if previous is None or self._evaluation_sort_key(evaluation) >= self._evaluation_sort_key(previous):
                evaluations_by_trigger[spec.trigger_id] = evaluation
                claims_by_trigger[spec.trigger_id] = claim
        return tuple(evaluations_by_trigger.values()), claims_by_trigger

    def _spec_for_claim(self, claim: FusedEventClaim) -> _FusedTriggerSpec | None:
        """Return the runtime mapping for one fused claim, if supported."""

        state = str(getattr(claim, "state", "") or "").strip().lower()
        if state == "possible_fall":
            return _FusedTriggerSpec(
                trigger_id="possible_fall",
                priority=SocialTriggerPriority.POSSIBLE_FALL,
                threshold=_FUSED_POSSIBLE_FALL_THRESHOLD,
                prompt="Brauchst du Hilfe?",
                reason=(
                    "Short-window multimodal fusion observed a recent fall-like transition "
                    "with continued low-floor support."
                ),
            )
        if state == "floor_stillness_after_drop":
            return _FusedTriggerSpec(
                trigger_id="floor_stillness",
                priority=SocialTriggerPriority.FLOOR_STILLNESS,
                threshold=_FUSED_FLOOR_STILLNESS_THRESHOLD,
                prompt=self._with_name(
                    base="Antworte mir kurz: Ist alles okay?",
                    with_name="Hey {name}, antworte mir kurz: Ist alles okay?",
                ),
                reason=(
                    "Short-window multimodal fusion observed floor-level stillness after "
                    "a recent drop-like transition."
                ),
            )
        if state in {"distress_possible", "cry_like_distress_possible"}:
            return _FusedTriggerSpec(
                trigger_id="distress_possible",
                priority=SocialTriggerPriority.DISTRESS_POSSIBLE,
                threshold=_FUSED_DISTRESS_THRESHOLD,
                prompt=self._with_name(
                    base="Ich wollte nur kurz fragen, ob alles in Ordnung ist.",
                    with_name="Hey {name}, ich wollte nur kurz fragen, ob alles in Ordnung ist.",
                ),
                reason=(
                    "Short-window multimodal fusion aligned distress-like audio with "
                    "recent concerning visible posture."
                ),
            )
        return None

    def _with_name(self, *, base: str, with_name: str) -> str:
        """Format one prompt variant with the configured display name."""

        if self._user_name is None:
            return base
        return with_name.format(name=self._user_name)

    def _evidence_from_claim(self, claim: FusedEventClaim) -> tuple[TriggerScoreEvidence, ...]:
        """Render one fused claim into bounded runtime evidence items."""

        audio_support = ",".join(claim.supporting_audio_events) or "none"
        vision_support = ",".join(claim.supporting_vision_events) or "none"
        blocked_by = ",".join(claim.blocked_by) or "none"
        return (
            TriggerScoreEvidence(
                key="fused_claim_confidence",
                value=claim.confidence,
                weight=1.0,
                detail=(
                    f"state={claim.state} confidence={claim.confidence:.3f} "
                    f"source={claim.source} action_level={claim.action_level.value}"
                ),
            ),
            TriggerScoreEvidence(
                key="supporting_audio_events",
                value=1.0 if claim.supporting_audio_events else 0.0,
                weight=0.0,
                detail=f"audio={audio_support}",
            ),
            TriggerScoreEvidence(
                key="supporting_vision_events",
                value=1.0 if claim.supporting_vision_events else 0.0,
                weight=0.0,
                detail=f"vision={vision_support}",
            ),
            TriggerScoreEvidence(
                key="delivery_policy",
                value=1.0 if claim.delivery_allowed else 0.0,
                weight=0.0,
                detail=f"delivery_allowed={claim.delivery_allowed} blocked_by={blocked_by}",
            ),
        )

    def _decision_from_evaluation(self, evaluation: SocialTriggerEvaluation) -> SocialTriggerDecision:
        """Convert one passing evaluation into the runtime decision contract."""

        return SocialTriggerDecision(
            trigger_id=evaluation.trigger_id,
            prompt=evaluation.prompt,
            reason=evaluation.reason,
            observed_at=evaluation.observed_at,
            priority=evaluation.priority,
            score=evaluation.score,
            threshold=evaluation.threshold,
            evidence=evaluation.evidence,
        )

    @staticmethod
    def _evaluation_sort_key(evaluation: SocialTriggerEvaluation) -> tuple[int, float, int]:
        """Return one stable ordering key for competing mapped claims."""

        return (1 if evaluation.passed else 0, evaluation.score, int(evaluation.priority))

    @staticmethod
    def _coerce_display_name(value: object) -> str | None:
        """Return one bounded display name or ``None`` when unset."""

        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text:
            return None
        return text[:120]


__all__ = ["SafetyTriggerFusionBridge"]
