"""Bridge short-window event fusion into the active runtime safety path.

This module keeps ``service.py`` orchestration-only. It composes the existing
``SocialTriggerEngine`` with the newer ``event_fusion`` tracker and prefers
fused safety claims when they are available, while preserving the legacy
engine as the fallback path for non-safety triggers and visibility-loss cases
that the current V1 fusion tracker does not yet model directly.
"""

# CHANGELOG: 2026-03-29
# BUG-1: Fusion-path exceptions no longer take down ``observe()``; the bridge now fails open to the surviving path.
# BUG-2: Fused claims now keep claim event-time semantics and are rejected when stale or replayed.
# BUG-3: Shared bridge state is synchronized for threaded and free-threaded runtimes.
# SEC-1: Fused claim payloads are bounded and sanitized to reduce easy CPU/log-memory denial of service on Raspberry Pi 4.
# SEC-2: Single-modality fused safety claims now require stricter runtime evidence before they can dispatch.
# IMP-1: Frontier 2026 upgrade: missing-modality, timing-drift, and asynchronous-event robustness for multimodal safety triggering.
# IMP-2: Added structured runtime diagnostics and conservative calibration hooks while preserving the public entrypoint.

from __future__ import annotations

import logging
import math
import re
import threading
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

LOGGER = logging.getLogger(__name__)

_FUSED_POSSIBLE_FALL_THRESHOLD = 0.72
_FUSED_FLOOR_STILLNESS_THRESHOLD = 0.74
_FUSED_DISTRESS_THRESHOLD = 0.70

_DEFAULT_MAX_FUSED_CLAIMS_PER_TICK = 16
_DEFAULT_MAX_SUPPORT_EVENTS_PER_MODALITY = 8
_DEFAULT_MAX_TEXT_CHARS = 240
_DEFAULT_MAX_SEEN_CLAIM_CACHE_SIZE = 256
_DEFAULT_SINGLE_MODALITY_THRESHOLD_BOOST = 0.08
_DEFAULT_UNKNOWN_MODALITY_THRESHOLD_BOOST = 0.14
_DEFAULT_MAX_CLAIM_AGE_BY_STATE = {
    "possible_fall": 4.0,
    "floor_stillness_after_drop": 15.0,
    "distress_possible": 8.0,
    "cry_like_distress_possible": 8.0,
}
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")


@dataclass(frozen=True, slots=True)
class _FusedTriggerSpec:
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
        self.config = config
        self.engine = engine
        self.tracker = tracker or MultimodalEventFusionTracker()
        self._lock = threading.RLock()

        self._last_fused_claims: tuple[FusedEventClaim, ...] = ()
        self._last_fused_evaluations: tuple[SocialTriggerEvaluation, ...] = ()
        self._last_fused_claim_by_trigger_id: dict[str, FusedEventClaim] = {}
        self._last_selected_source: str | None = None
        self._last_selected_claim: FusedEventClaim | None = None
        self._last_observed_at: float | None = None
        self._seen_claim_first_observed_at: dict[str, float] = {}
        self._seen_claim_last_observed_at: dict[str, float] = {}

        self._max_fused_claims_per_tick = self._cfg_int(
            "safety_trigger_fusion_max_claims_per_tick", _DEFAULT_MAX_FUSED_CLAIMS_PER_TICK, 1, 64
        )
        self._max_support_events_per_modality = self._cfg_int(
            "safety_trigger_fusion_max_support_events_per_modality", _DEFAULT_MAX_SUPPORT_EVENTS_PER_MODALITY, 1, 32
        )
        self._max_text_chars = self._cfg_int(
            "safety_trigger_fusion_max_text_chars", _DEFAULT_MAX_TEXT_CHARS, 32, 1024
        )
        self._max_seen_claim_cache_size = self._cfg_int(
            "safety_trigger_fusion_max_seen_claim_cache_size", _DEFAULT_MAX_SEEN_CLAIM_CACHE_SIZE, 32, 2048
        )
        self._single_modality_threshold_boost = self._cfg_float(
            "safety_trigger_fusion_single_modality_threshold_boost",
            _DEFAULT_SINGLE_MODALITY_THRESHOLD_BOOST,
            0.0,
            0.30,
        )
        self._unknown_modality_threshold_boost = self._cfg_float(
            "safety_trigger_fusion_unknown_modality_threshold_boost",
            _DEFAULT_UNKNOWN_MODALITY_THRESHOLD_BOOST,
            0.0,
            0.40,
        )
        self._max_claim_age_by_state = {
            state: self._cfg_float(f"safety_trigger_fusion_max_age_seconds_{state}", default, 0.1, 300.0)
            for state, default in _DEFAULT_MAX_CLAIM_AGE_BY_STATE.items()
        }
        self._user_name = self._coerce_display_name(
            getattr(engine, "user_name", None) or getattr(config, "user_display_name", None)
        )

    @classmethod
    def from_config(
        cls,
        config: TwinrConfig,
        *,
        engine: SocialTriggerEngine | object,
    ) -> "SafetyTriggerFusionBridge":
        return cls(config=config, engine=engine)

    @property
    def best_evaluation(self) -> SocialTriggerEvaluation | None:
        with self._lock:
            engine_best = self._safe_engine_best_evaluation()
            fused_best = self.best_fused_evaluation
            candidates = tuple(item for item in (engine_best, fused_best) if item is not None)
            if not candidates:
                return None
            passed = tuple(item for item in candidates if item.passed)
            return max(
                passed or candidates,
                key=lambda item: (int(item.priority), item.score) if item.passed else (0, item.score),
            )

    @property
    def best_fused_evaluation(self) -> SocialTriggerEvaluation | None:
        with self._lock:
            if not self._last_fused_evaluations:
                return None
            passed = tuple(item for item in self._last_fused_evaluations if item.passed)
            if passed:
                return max(passed, key=lambda item: (int(item.priority), item.score))
            return max(self._last_fused_evaluations, key=lambda item: (item.score, int(item.priority)))

    @property
    def last_fused_claims(self) -> tuple[FusedEventClaim, ...]:
        with self._lock:
            return self._last_fused_claims

    @property
    def last_fused_evaluations(self) -> tuple[SocialTriggerEvaluation, ...]:
        with self._lock:
            return self._last_fused_evaluations

    @property
    def last_selected_source(self) -> str | None:
        with self._lock:
            return self._last_selected_source

    @property
    def last_selected_claim(self) -> FusedEventClaim | None:
        with self._lock:
            return self._last_selected_claim

    def observe(
        self,
        observation: SocialObservation,
        *,
        room_busy_or_overlapping: bool = False,
    ) -> SocialTriggerDecision | None:
        with self._lock:
            observed_at = self._coerce_timestamp(getattr(observation, "observed_at", None), self._last_observed_at)
            if observed_at is not None:
                self._last_observed_at = observed_at

            engine_decision = self._observe_engine_safely(observation)
            self._last_fused_claims = self._observe_tracker_safely(
                observation, room_busy_or_overlapping=room_busy_or_overlapping
            )
            self._last_fused_evaluations, self._last_fused_claim_by_trigger_id = self._evaluate_fused_claims_safely(
                observed_at=observed_at, claims=self._last_fused_claims
            )

            fused_eval = self.best_fused_evaluation
            fused_decision = None
            if fused_eval is not None and fused_eval.passed:
                fused_decision = self._decision_from_evaluation(fused_eval)

            selected = self._select_runtime_decision(
                engine_decision=engine_decision,
                fused_decision=fused_decision,
            )
            if selected is fused_decision and selected is not None:
                note_dispatched = getattr(self.engine, "note_trigger_dispatched", None)
                if callable(note_dispatched):
                    try:
                        note_dispatched(selected.trigger_id, observed_at=selected.observed_at)
                    except Exception:
                        LOGGER.exception(
                            "SafetyTriggerFusionBridge note_trigger_dispatched failed for trigger_id=%s",
                            selected.trigger_id,
                        )
            self._prune_seen_claim_cache(observed_at)
            return selected

    def _observe_engine_safely(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        observe = getattr(self.engine, "observe", None)
        if not callable(observe):
            return None
        try:
            result = observe(observation)
        except Exception:
            LOGGER.exception("SafetyTriggerFusionBridge legacy engine.observe failed")
            return None
        if result is None or isinstance(result, SocialTriggerDecision):
            return result
        LOGGER.warning(
            "SafetyTriggerFusionBridge ignored unexpected engine decision type=%s",
            type(result).__name__,
        )
        return None

    def _observe_tracker_safely(
        self,
        observation: SocialObservation,
        *,
        room_busy_or_overlapping: bool,
    ) -> tuple[FusedEventClaim, ...]:
        try:
            claims = self.tracker.observe(observation, room_busy_or_overlapping=room_busy_or_overlapping)
        except Exception:
            LOGGER.exception("SafetyTriggerFusionBridge fusion tracker.observe failed")
            return ()
        if not claims:
            return ()
        try:
            iterator = iter(claims)
        except TypeError:
            LOGGER.warning(
                "SafetyTriggerFusionBridge ignored non-iterable fused claims payload type=%s",
                type(claims).__name__,
            )
            return ()

        accepted: list[FusedEventClaim] = []
        for claim in iterator:
            if isinstance(claim, FusedEventClaim):
                accepted.append(claim)
            else:
                LOGGER.warning(
                    "SafetyTriggerFusionBridge ignored unexpected fused claim type=%s",
                    type(claim).__name__,
                )
            if len(accepted) >= self._max_fused_claims_per_tick:
                LOGGER.warning(
                    "SafetyTriggerFusionBridge capped fused claims at %s for one observation tick",
                    self._max_fused_claims_per_tick,
                )
                break
        return tuple(accepted)

    def _select_runtime_decision(
        self,
        *,
        engine_decision: SocialTriggerDecision | None,
        fused_decision: SocialTriggerDecision | None,
    ) -> SocialTriggerDecision | None:
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

    def _evaluate_fused_claims_safely(
        self,
        *,
        observed_at: float | None,
        claims: tuple[FusedEventClaim, ...],
    ) -> tuple[tuple[SocialTriggerEvaluation, ...], dict[str, FusedEventClaim]]:
        try:
            return self._evaluate_fused_claims(observed_at=observed_at, claims=claims)
        except Exception:
            LOGGER.exception("SafetyTriggerFusionBridge fused claim evaluation failed")
            return (), {}

    def _evaluate_fused_claims(
        self,
        *,
        observed_at: float | None,
        claims: tuple[FusedEventClaim, ...],
    ) -> tuple[tuple[SocialTriggerEvaluation, ...], dict[str, FusedEventClaim]]:
        engine_evaluations = {item.trigger_id: item for item in self._safe_engine_last_evaluations()}
        evaluations_by_trigger: dict[str, SocialTriggerEvaluation] = {}
        claims_by_trigger: dict[str, FusedEventClaim] = {}

        for claim in claims:
            spec = self._spec_for_claim(claim)
            if spec is None:
                continue

            state = self._sanitize_text(getattr(claim, "state", None), "unknown_state")
            audio_events = self._tokens(
                getattr(claim, "supporting_audio_events", ()),
                max_items=self._max_support_events_per_modality,
            )
            vision_events = self._tokens(
                getattr(claim, "supporting_vision_events", ()),
                max_items=self._max_support_events_per_modality,
            )
            modalities = self._infer_modalities(claim, audio_events, vision_events)
            threshold = self._threshold_for_claim(spec.trigger_id, spec.threshold, modalities)

            signature = self._claim_signature(claim, audio_events, vision_events)
            claim_observed_at = self._resolve_claim_observed_at(
                claim_signature=signature,
                observed_at=observed_at,
                extracted_claim_observed_at=self._extract_claim_observed_at(claim),
            )
            claim_age_seconds = (
                None
                if observed_at is None or claim_observed_at is None
                else max(0.0, observed_at - claim_observed_at)
            )
            claim_skew_seconds = (
                None
                if observed_at is None or claim_observed_at is None
                else max(0.0, claim_observed_at - observed_at)
            )

            blocked_reason = None
            if claim_observed_at is None:
                blocked_reason = "claim_missing_timestamp"
            elif claim_skew_seconds is not None and claim_skew_seconds > 1.0:
                blocked_reason = "claim_clock_skew"
            elif (
                claim_age_seconds is not None
                and claim_age_seconds
                > self._max_claim_age_by_state.get(state, max(self._max_claim_age_by_state.values()))
            ):
                blocked_reason = "claim_stale"
            elif not bool(getattr(claim, "active", True)):
                blocked_reason = "claim_inactive"
            elif not bool(getattr(claim, "delivery_allowed", True)):
                blocked_reason = "claim_blocked_" + "_".join(
                    self._tokens(getattr(claim, "blocked_by", ()), default=("policy",))
                )
            elif (
                engine_evaluations.get(spec.trigger_id) is not None
                and engine_evaluations[spec.trigger_id].blocked_reason == "cooldown_active"
            ):
                blocked_reason = "cooldown_active"

            confidence = self._coerce_confidence(getattr(claim, "confidence", None))
            evaluation = SocialTriggerEvaluation(
                trigger_id=spec.trigger_id,
                prompt=spec.prompt,
                reason=spec.reason,
                # BREAKING: cooldown and provenance now follow the claim event time when available.
                observed_at=claim_observed_at,
                priority=spec.priority,
                score=confidence,
                # BREAKING: single-modality or unknown-modality claims require stricter thresholds.
                threshold=threshold,
                evidence=self._evidence_from_claim(
                    claim=claim,
                    audio_events=audio_events,
                    vision_events=vision_events,
                    modalities=modalities,
                    claim_age_seconds=claim_age_seconds,
                    runtime_threshold=threshold,
                ),
                passed=(blocked_reason is None and confidence >= threshold),
                blocked_reason=blocked_reason,
            )
            previous = evaluations_by_trigger.get(spec.trigger_id)
            if previous is None or self._evaluation_sort_key(evaluation) >= self._evaluation_sort_key(previous):
                evaluations_by_trigger[spec.trigger_id] = evaluation
                claims_by_trigger[spec.trigger_id] = claim

        return tuple(evaluations_by_trigger.values()), claims_by_trigger

    def _spec_for_claim(self, claim: FusedEventClaim) -> _FusedTriggerSpec | None:
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
        return base if self._user_name is None else with_name.format(name=self._user_name)

    def _evidence_from_claim(
        self,
        *,
        claim: FusedEventClaim,
        audio_events: tuple[str, ...],
        vision_events: tuple[str, ...],
        modalities: tuple[str, ...],
        claim_age_seconds: float | None,
        runtime_threshold: float,
    ) -> tuple[TriggerScoreEvidence, ...]:
        source = self._sanitize_text(getattr(claim, "source", None), "unknown")
        state = self._sanitize_text(getattr(claim, "state", None), "unknown")
        action_level = self._sanitize_text(
            getattr(getattr(claim, "action_level", None), "value", getattr(claim, "action_level", None)),
            "unknown",
        )
        blocked_by = ",".join(self._tokens(getattr(claim, "blocked_by", ()), default=("none",)))
        confidence = self._coerce_confidence(getattr(claim, "confidence", None))
        return (
            TriggerScoreEvidence(
                key="fused_claim_confidence",
                value=confidence,
                weight=1.0,
                detail=f"state={state} confidence={confidence:.3f} source={source} action_level={action_level}",
            ),
            TriggerScoreEvidence(
                key="supporting_audio_events",
                value=1.0 if audio_events else 0.0,
                weight=0.0,
                detail=f"audio={','.join(audio_events) or 'none'}",
            ),
            TriggerScoreEvidence(
                key="supporting_vision_events",
                value=1.0 if vision_events else 0.0,
                weight=0.0,
                detail=f"vision={','.join(vision_events) or 'none'}",
            ),
            TriggerScoreEvidence(
                key="corroborating_modalities",
                value=float(len(modalities)),
                weight=0.0,
                detail=f"modalities={','.join(modalities) or 'none'}",
            ),
            TriggerScoreEvidence(
                key="claim_age_seconds",
                value=-1.0 if claim_age_seconds is None else claim_age_seconds,
                weight=0.0,
                detail=(
                    "claim_age_seconds="
                    f"{'unknown' if claim_age_seconds is None else f'{claim_age_seconds:.3f}'}"
                ),
            ),
            TriggerScoreEvidence(
                key="runtime_threshold",
                value=runtime_threshold,
                weight=0.0,
                detail=f"runtime_threshold={runtime_threshold:.3f}",
            ),
            TriggerScoreEvidence(
                key="delivery_policy",
                value=1.0 if bool(getattr(claim, "delivery_allowed", True)) else 0.0,
                weight=0.0,
                detail=(
                    f"delivery_allowed={bool(getattr(claim, 'delivery_allowed', True))} "
                    f"blocked_by={blocked_by}"
                ),
            ),
        )

    @staticmethod
    def _decision_from_evaluation(evaluation: SocialTriggerEvaluation) -> SocialTriggerDecision:
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

    def _threshold_for_claim(
        self,
        trigger_id: str,
        base_threshold: float,
        modalities: tuple[str, ...],
    ) -> float:
        threshold = base_threshold
        if trigger_id in {"possible_fall", "distress_possible"}:
            if len(modalities) == 0:
                threshold += self._unknown_modality_threshold_boost
            elif len(modalities) == 1:
                threshold += self._single_modality_threshold_boost
        return max(0.0, min(1.0, threshold))

    def _safe_engine_best_evaluation(self) -> SocialTriggerEvaluation | None:
        try:
            value = getattr(self.engine, "best_evaluation", None)
        except Exception:
            LOGGER.exception("SafetyTriggerFusionBridge reading engine.best_evaluation failed")
            return None
        return value if isinstance(value, SocialTriggerEvaluation) else None

    def _safe_engine_last_evaluations(self) -> tuple[SocialTriggerEvaluation, ...]:
        try:
            raw = getattr(self.engine, "last_evaluations", ()) or ()
        except Exception:
            LOGGER.exception("SafetyTriggerFusionBridge reading engine.last_evaluations failed")
            return ()
        return tuple(item for item in tuple(raw) if isinstance(item, SocialTriggerEvaluation))

    def _extract_claim_observed_at(self, claim: FusedEventClaim) -> float | None:
        for name in (
            "observed_at",
            "event_time",
            "event_at",
            "window_end_at",
            "window_ended_at",
            "updated_at",
            "created_at",
            "timestamp",
        ):
            value = self._coerce_timestamp(getattr(claim, name, None), None)
            if value is not None:
                return value
        return None

    def _claim_signature(
        self,
        claim: FusedEventClaim,
        audio_events: tuple[str, ...],
        vision_events: tuple[str, ...],
    ) -> str:
        state = self._sanitize_text(getattr(claim, "state", None), "unknown")
        source = self._sanitize_text(getattr(claim, "source", None), "unknown")
        return f"{state}|{source}|{','.join(audio_events) or 'none'}|{','.join(vision_events) or 'none'}"

    def _resolve_claim_observed_at(
        self,
        *,
        claim_signature: str,
        observed_at: float | None,
        extracted_claim_observed_at: float | None,
    ) -> float | None:
        if extracted_claim_observed_at is not None:
            self._seen_claim_first_observed_at[claim_signature] = extracted_claim_observed_at
            self._seen_claim_last_observed_at[claim_signature] = observed_at or extracted_claim_observed_at
            return extracted_claim_observed_at
        if observed_at is None:
            return None
        first_seen = self._seen_claim_first_observed_at.setdefault(claim_signature, observed_at)
        self._seen_claim_last_observed_at[claim_signature] = observed_at
        return first_seen

    def _prune_seen_claim_cache(self, observed_at: float | None) -> None:
        if not self._seen_claim_first_observed_at:
            return
        max_age = max(self._max_claim_age_by_state.values())
        if observed_at is not None:
            stale_before = observed_at - max_age - 5.0
            stale_keys = [
                key
                for key, last_seen in tuple(self._seen_claim_last_observed_at.items())
                if last_seen < stale_before
            ]
            for key in stale_keys:
                self._seen_claim_last_observed_at.pop(key, None)
                self._seen_claim_first_observed_at.pop(key, None)
        if len(self._seen_claim_first_observed_at) <= self._max_seen_claim_cache_size:
            return
        for key, _ in sorted(self._seen_claim_last_observed_at.items(), key=lambda item: item[1])[
            : len(self._seen_claim_first_observed_at) - self._max_seen_claim_cache_size
        ]:
            self._seen_claim_last_observed_at.pop(key, None)
            self._seen_claim_first_observed_at.pop(key, None)

    def _infer_modalities(
        self,
        claim: FusedEventClaim,
        audio_events: tuple[str, ...],
        vision_events: tuple[str, ...],
    ) -> tuple[str, ...]:
        result: list[str] = []
        if audio_events:
            result.append("audio")
        if vision_events:
            result.append("vision")
        if result:
            return tuple(result)
        source = self._sanitize_text(getattr(claim, "source", None), "").lower()
        if any(token in source for token in ("audio", "microphone", "mic", "speech", "cry", "sound")):
            result.append("audio")
        if any(token in source for token in ("vision", "camera", "video", "pose", "image", "rgb")):
            result.append("vision")
        return tuple(dict.fromkeys(result))

    def _tokens(
        self,
        value: object,
        *,
        default: tuple[str, ...] = (),
        max_items: int | None = None,
    ) -> tuple[str, ...]:
        if max_items is None:
            max_items = self._max_support_events_per_modality
        if isinstance(value, str):
            items = (value,)
        elif isinstance(value, (tuple, list, set, frozenset)):
            items = tuple(value)
        else:
            items = ()
        out: list[str] = []
        for item in items:
            text = self._sanitize_text(item, "")
            if text:
                out.append(text)
            if len(out) >= max_items:
                break
        return tuple(out) or default

    def _sanitize_text(self, value: object, default: str) -> str:
        if not isinstance(value, str):
            return default
        text = _CONTROL_CHARS_RE.sub(" ", value)
        text = " ".join(text.split()).strip()
        return text[: self._max_text_chars] if text else default

    def _coerce_display_name(self, value: object) -> str | None:
        text = self._sanitize_text(value, "")
        return text[:120] if text else None

    @staticmethod
    def _coerce_timestamp(value: object, fallback: float | None) -> float | None:
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return fallback

    @staticmethod
    def _coerce_confidence(value: object) -> float:
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return max(0.0, min(1.0, value))
        return 0.0

    @staticmethod
    def _evaluation_sort_key(evaluation: SocialTriggerEvaluation) -> tuple[int, float, int]:
        return (1 if evaluation.passed else 0, evaluation.score, int(evaluation.priority))

    def _cfg_int(self, name: str, default: int, minimum: int, maximum: int) -> int:
        value = getattr(self.config, name, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return max(minimum, min(maximum, int(value)))
        return default

    def _cfg_float(self, name: str, default: float, minimum: float, maximum: float) -> float:
        value = getattr(self.config, name, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return max(minimum, min(maximum, value))
        return default


__all__ = ["SafetyTriggerFusionBridge"]
