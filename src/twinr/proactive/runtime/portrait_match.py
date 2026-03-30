# CHANGELOG: 2026-03-29
# BUG-1: Missing/partial camera facts no longer collapse into "no_visible_person"; only an explicit false
#         visibility signal or zero detected live faces suppresses portrait matching.
# BUG-2: Fixed checked_age_s computation across mixed clock domains (monotonic vs wall-clock), which could
#         previously mark stale observations as fresh and silently leak stale identity claims downstream.
# BUG-3: Removed fabricated medium-confidence fallbacks when the provider omits calibrated confidence; such
#         positives are now downgraded to confirm-first instead of being reported as moderately trustworthy.
# SEC-1: Added liveness / presentation-attack gating so practical print/replay spoof signals can block
#         personalization on a Raspberry Pi deployment when the upstream pipeline exposes PAD evidence.
# SEC-2: Ambiguous, stale, low-quality, and otherwise unsafe results now redact matched identity fields by
#         default, reducing low-confidence identity leakage into automation facts and logs.
# IMP-1: Added freshness, FIQA/quality, open-set uncertainty, and liveness-aware claim fusion aligned with
#         2026 edge-face-ID runtime patterns.
# IMP-2: Added schema-tolerant normalization and configurable policy thresholds so newer providers/backends
#         can project richer evidence without rewriting this policy layer.

"""Project local portrait-match observations into conservative, uncertainty-aware runtime claims."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from twinr.hardware.portrait_match import PortraitMatchProvider
from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_mapping,
    coerce_optional_bool,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_ratio,
    normalize_text,
)


_LOG = logging.getLogger(__name__)

_STABLE_TEMPORAL_STATES = {"stable", "stable_match", "stable_other_user", "stable_nonmatch"}
_LOW_QUALITY_STATES = {
    "bad",
    "blurred",
    "dark",
    "low",
    "low_quality",
    "occluded",
    "poor",
    "too_dark",
    "too_small",
}
_LIVENESS_ATTACK_STATES = {
    "attack",
    "mask_attack",
    "non_live",
    "non_live_face",
    "presentation_attack",
    "presentation_attack_detected",
    "print_attack",
    "replay_attack",
    "spoof",
    "spoof_detected",
}
_LIVENESS_FAIL_STATES = {"fail", "failed", "not_live", "rejected", "uncertain"}
_LIVENESS_PASS_STATES = {"bonafide", "genuine", "live", "ok", "passed"}
_CANONICAL_STATE_ALIASES = {
    "likely_reference_user": {
        "likely_reference_user",
        "main_user_match",
        "matched_reference_user",
        "match",
        "reference_match",
        "reference_user",
        "reference_user_match",
    },
    "known_other_user": {
        "known_other_user",
        "matched_other_user",
        "other_enrolled_user",
        "other_user_match",
    },
    "ambiguous_identity": {
        "ambiguous_identity",
        "identity_collision",
        "multiple_candidates",
        "multiple_close_candidates",
    },
    "uncertain_match": {
        "low_confidence_match",
        "needs_confirmation",
        "uncertain",
        "uncertain_match",
        "uncertain_reference_user",
        "unconfirmed_match",
    },
    "provider_unavailable": {
        "provider_error",
        "provider_unavailable",
        "unavailable",
    },
    "no_visible_person": {
        "no_face",
        "no_face_detected",
        "no_live_face",
        "no_person",
        "no_visible_person",
    },
    "blocked_presentation_attack": {
        "blocked_presentation_attack",
        "mask_attack",
        "non_live_face",
        "presentation_attack",
        "presentation_attack_detected",
        "print_attack",
        "replay_attack",
        "spoof",
        "spoof_detected",
    },
}


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="local_portrait_match",
        requires_confirmation=True,
    )


@dataclass(frozen=True, slots=True)
class PortraitMatchPolicy:
    """Conservative runtime policy used to project provider outputs into claims."""

    max_reference_observation_age_s: float = 3.0
    max_nonmatch_observation_age_s: float = 5.0
    min_reference_confidence: float = 0.72
    min_other_user_confidence: float = 0.68
    min_quality_score: float = 0.45
    min_liveness_confidence: float = 0.60
    max_open_set_risk: float = 0.35
    redact_identity_when_uncertain: bool = True

    @classmethod
    def from_fact_map(cls, value: object | None) -> "PortraitMatchPolicy":
        payload = coerce_mapping(value)
        if not payload:
            return cls()

        reference_age = _coalesce_non_negative_float(payload.get("max_reference_observation_age_s"))
        nonmatch_age = _coalesce_non_negative_float(payload.get("max_nonmatch_observation_age_s"))
        reference_conf = _coalesce_score(payload.get("min_reference_confidence"))
        other_conf = _coalesce_score(payload.get("min_other_user_confidence"))
        quality = _coalesce_score(payload.get("min_quality_score"))
        liveness = _coalesce_score(payload.get("min_liveness_confidence"))
        open_set = _coalesce_score(payload.get("max_open_set_risk"))
        redact = coerce_optional_bool(payload.get("redact_identity_when_uncertain"))

        return cls(
            max_reference_observation_age_s=(
                reference_age if reference_age is not None else cls.max_reference_observation_age_s
            ),
            max_nonmatch_observation_age_s=(
                nonmatch_age if nonmatch_age is not None else cls.max_nonmatch_observation_age_s
            ),
            min_reference_confidence=(
                reference_conf if reference_conf is not None else cls.min_reference_confidence
            ),
            min_other_user_confidence=(
                other_conf if other_conf is not None else cls.min_other_user_confidence
            ),
            min_quality_score=quality if quality is not None else cls.min_quality_score,
            min_liveness_confidence=(
                liveness if liveness is not None else cls.min_liveness_confidence
            ),
            max_open_set_risk=open_set if open_set is not None else cls.max_open_set_risk,
            redact_identity_when_uncertain=(
                redact if redact is not None else cls.redact_identity_when_uncertain
            ),
        )


@dataclass(frozen=True, slots=True)
class _EvidenceSignals:
    quality_score: float | None = None
    quality_state: str | None = None
    liveness_confidence: float | None = None
    liveness_state: str | None = None
    presentation_attack_detected: bool | None = None
    open_set_risk: float | None = None
    gallery_uncertainty: float | None = None
    embedding_uncertainty: float | None = None


@dataclass(frozen=True, slots=True)
class PortraitMatchSnapshot:
    """Describe whether current live capture matches the stored main-user portrait."""

    observed_at: float | None = None
    state: str = "provider_unavailable"
    matches_reference_user: bool = False
    policy_recommendation: str = "unavailable"
    block_reason: str | None = None
    claim: RuntimeClaimMetadata = field(default_factory=_default_claim)
    similarity_score: float | None = None
    checked_age_s: float | None = None
    live_face_count: int | None = None
    reference_face_count: int | None = None
    reference_image_count: int | None = None
    matched_user_id: str | None = None
    matched_user_display_name: str | None = None
    candidate_user_count: int | None = None
    temporal_state: str | None = None
    temporal_observation_count: int | None = None
    fused_confidence: float | None = None
    backend_name: str | None = None
    observation_fresh: bool | None = None
    freshness_limit_s: float | None = None
    freshness_clock: str | None = None
    quality_score: float | None = None
    quality_state: str | None = None
    liveness_confidence: float | None = None
    liveness_state: str | None = None
    presentation_attack_detected: bool | None = None
    open_set_risk: float | None = None
    gallery_uncertainty: float | None = None
    embedding_uncertainty: float | None = None

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the portrait-match snapshot into automation facts."""

        payload = {
            "observed_at": self.observed_at,
            "state": self.state,
            "matches_reference_user": self.matches_reference_user,
            "policy_recommendation": self.policy_recommendation,
            "block_reason": self.block_reason,
            "similarity_score": self.similarity_score,
            "checked_age_s": self.checked_age_s,
            "live_face_count": self.live_face_count,
            "reference_face_count": self.reference_face_count,
            "reference_image_count": self.reference_image_count,
            "matched_user_id": self.matched_user_id,
            "matched_user_display_name": self.matched_user_display_name,
            "candidate_user_count": self.candidate_user_count,
            "temporal_state": self.temporal_state,
            "temporal_observation_count": self.temporal_observation_count,
            "fused_confidence": self.fused_confidence,
            "backend_name": self.backend_name,
            "observation_fresh": self.observation_fresh,
            "freshness_limit_s": self.freshness_limit_s,
            "freshness_clock": self.freshness_clock,
            "quality_score": self.quality_score,
            "quality_state": self.quality_state,
            "liveness_confidence": self.liveness_confidence,
            "liveness_state": self.liveness_state,
            "presentation_attack_detected": self.presentation_attack_detected,
            "open_set_risk": self.open_set_risk,
            "gallery_uncertainty": self.gallery_uncertainty,
            "embedding_uncertainty": self.embedding_uncertainty,
        }
        payload.update(self.claim.to_payload())
        return payload

    def event_data(self) -> dict[str, object]:
        """Serialize the portrait-match snapshot into compact flat event fields."""

        return {
            "portrait_match_state": self.state,
            "portrait_match_matches_reference_user": self.matches_reference_user,
            "portrait_match_confidence": self.claim.confidence,
            "portrait_match_policy": self.policy_recommendation,
            "portrait_match_temporal_state": self.temporal_state,
            "portrait_match_quality_state": self.quality_state,
            "portrait_match_liveness_state": self.liveness_state,
        }

    @classmethod
    def from_fact_map(
        cls,
        value: object | None,
    ) -> "PortraitMatchSnapshot | None":
        """Parse one serialized portrait-match payload."""

        payload = coerce_mapping(value)
        if not payload:
            return None
        return cls(
            observed_at=coerce_optional_float(payload.get("observed_at")),
            state=normalize_text(payload.get("state")) or "provider_unavailable",
            matches_reference_user=coerce_optional_bool(payload.get("matches_reference_user")) is True,
            policy_recommendation=normalize_text(payload.get("policy_recommendation")) or "unavailable",
            block_reason=normalize_text(payload.get("block_reason")) or None,
            claim=RuntimeClaimMetadata.from_payload(
                payload,
                default_source="local_portrait_match",
                default_requires_confirmation=True,
            ),
            similarity_score=coerce_optional_float(payload.get("similarity_score")),
            checked_age_s=coerce_optional_float(payload.get("checked_age_s")),
            live_face_count=coerce_optional_int(payload.get("live_face_count")),
            reference_face_count=coerce_optional_int(payload.get("reference_face_count")),
            reference_image_count=coerce_optional_int(payload.get("reference_image_count")),
            matched_user_id=normalize_text(payload.get("matched_user_id")) or None,
            matched_user_display_name=normalize_text(payload.get("matched_user_display_name")) or None,
            candidate_user_count=coerce_optional_int(payload.get("candidate_user_count")),
            temporal_state=normalize_text(payload.get("temporal_state")) or None,
            temporal_observation_count=coerce_optional_int(payload.get("temporal_observation_count")),
            fused_confidence=_coalesce_score(payload.get("fused_confidence")),
            backend_name=normalize_text(payload.get("backend_name")) or None,
            observation_fresh=coerce_optional_bool(payload.get("observation_fresh")),
            freshness_limit_s=coerce_optional_float(payload.get("freshness_limit_s")),
            freshness_clock=normalize_text(payload.get("freshness_clock")) or None,
            quality_score=_coalesce_score(payload.get("quality_score")),
            quality_state=normalize_text(payload.get("quality_state")) or None,
            liveness_confidence=_coalesce_score(payload.get("liveness_confidence")),
            liveness_state=normalize_text(payload.get("liveness_state")) or None,
            presentation_attack_detected=coerce_optional_bool(
                payload.get("presentation_attack_detected")
            ),
            open_set_risk=_coalesce_score(payload.get("open_set_risk")),
            gallery_uncertainty=_coalesce_score(payload.get("gallery_uncertainty")),
            embedding_uncertainty=_coalesce_score(payload.get("embedding_uncertainty")),
        )


def derive_portrait_match(
    *,
    observed_at: float | None,
    live_facts: dict[str, object] | object,
    provider: PortraitMatchProvider | None,
    ambiguous_room_guard: AmbiguousRoomGuardSnapshot | None = None,
    now_monotonic: float | None = None,
) -> PortraitMatchSnapshot:
    """Return one conservative portrait-match runtime snapshot."""

    facts = coerce_mapping(live_facts)
    policy = PortraitMatchPolicy.from_fact_map(facts.get("portrait_match_policy"))

    guard = ambiguous_room_guard or AmbiguousRoomGuardSnapshot.from_fact_map(
        facts.get("ambiguous_room_guard"),
    ) or derive_ambiguous_room_guard(
        observed_at=observed_at,
        live_facts=facts,
    )
    if guard.guard_active:
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state="blocked_ambiguous_room",
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason=guard.reason,
            claim=RuntimeClaimMetadata(
                confidence=guard.claim.confidence,
                source="local_portrait_match_plus_ambiguous_room_guard",
                requires_confirmation=True,
            ),
            backend_name=None if provider is None else _provider_name(provider),
        )

    camera = coerce_mapping(facts.get("camera"))
    person_visible = _camera_person_visible(camera)
    if person_visible is False:
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state="no_visible_person",
            matches_reference_user=False,
            policy_recommendation="unavailable",
            block_reason="no_visible_person",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="local_portrait_match",
                requires_confirmation=True,
            ),
            backend_name=None if provider is None else _provider_name(provider),
        )

    if provider is None:
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state="provider_unavailable",
            matches_reference_user=False,
            policy_recommendation="unavailable",
            block_reason="provider_unavailable",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="local_portrait_match",
                requires_confirmation=True,
            ),
        )

    try:
        observation = provider.observe()
    except Exception:
        _LOG.exception("PortraitMatchProvider.observe() failed")
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state="provider_unavailable",
            matches_reference_user=False,
            policy_recommendation="unavailable",
            block_reason="provider_error",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="local_portrait_match",
                requires_confirmation=True,
            ),
            backend_name=_provider_name(provider),
        )

    if observation is None:
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state="provider_unavailable",
            matches_reference_user=False,
            policy_recommendation="unavailable",
            block_reason="provider_empty_result",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="local_portrait_match",
                requires_confirmation=True,
            ),
            backend_name=_provider_name(provider),
        )

    backend_name = normalize_text(getattr(observation, "backend_name", None)) or _provider_name(provider)
    canonical_state = _canonicalize_observation_state(getattr(observation, "state", None))
    temporal_state = normalize_text(getattr(observation, "temporal_state", None)) or None
    checked_age_s, freshness_clock = _estimate_checked_age_s(
        checked_at=getattr(observation, "checked_at", None),
        observed_at=observed_at,
        now_monotonic=now_monotonic,
    )
    freshness_limit_s = _freshness_limit_s(canonical_state, policy)
    observation_fresh = None if checked_age_s is None else checked_age_s <= freshness_limit_s

    signals = _extract_evidence_signals(facts=facts, observation=observation)

    if observation_fresh is False:
        # BREAKING: stale provider observations are now downgraded to a dedicated state
        # instead of being treated as current evidence.
        return _build_snapshot(
            observed_at=observed_at,
            state="stale_observation",
            matches_reference_user=False,
            policy_recommendation="confirm_first",
            block_reason="stale_observation",
            claim_confidence=0.0,
            claim_source="local_portrait_match_staleness_guard",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    if canonical_state == "blocked_presentation_attack" or _is_presentation_attack(signals):
        # BREAKING: a practical spoof/PAD failure is now surfaced explicitly instead of
        # being flattened into a generic ambiguous/unavailable state.
        return _build_snapshot(
            observed_at=observed_at,
            state="blocked_presentation_attack",
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason="presentation_attack_detected",
            claim_confidence=0.0,
            claim_source="local_portrait_match_liveness_guard",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    if _observation_has_no_visible_face(observation, canonical_state):
        return _build_snapshot(
            observed_at=observed_at,
            state="no_visible_person",
            matches_reference_user=False,
            policy_recommendation="unavailable",
            block_reason="no_visible_person",
            claim_confidence=0.0,
            claim_source="local_portrait_match",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
        )

    if _is_low_quality(signals, policy):
        # BREAKING: low-quality face evidence is now surfaced explicitly rather than passing
        # through the provider state machine unchanged.
        return _build_snapshot(
            observed_at=observed_at,
            state="low_quality_observation",
            matches_reference_user=False,
            policy_recommendation="confirm_first",
            block_reason="face_quality_too_low",
            claim_confidence=0.0,
            claim_source="local_portrait_match_quality_guard",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    identity_confidence = _effective_identity_confidence(
        observation=observation,
        signals=signals,
        temporal_state=temporal_state,
    )
    candidate_user_count = coerce_optional_int(getattr(observation, "candidate_user_count", None))

    open_set_block_reason = _open_set_block_reason(
        state=canonical_state,
        signals=signals,
        candidate_user_count=candidate_user_count,
        temporal_state=temporal_state,
        policy=policy,
    )
    if open_set_block_reason is not None:
        return _build_snapshot(
            observed_at=observed_at,
            state="ambiguous_identity",
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason=open_set_block_reason,
            claim_confidence=identity_confidence if identity_confidence is not None else 0.0,
            claim_source="local_portrait_match_uncertainty_guard",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    if _has_liveness_shortfall(signals=signals, policy=policy):
        return _build_snapshot(
            observed_at=observed_at,
            state="uncertain_match",
            matches_reference_user=False,
            policy_recommendation="confirm_first",
            block_reason="liveness_insufficient",
            claim_confidence=0.0,
            claim_source="local_portrait_match_liveness_guard",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    if canonical_state == "likely_reference_user":
        # BREAKING: missing calibrated confidence no longer falls back to a fabricated
        # medium-confidence prior. Uncalibrated positives now require confirmation.
        if identity_confidence is None or identity_confidence < policy.min_reference_confidence:
            return _build_snapshot(
                observed_at=observed_at,
                state="uncertain_match",
                matches_reference_user=False,
                policy_recommendation="confirm_first",
                block_reason="insufficient_reference_confidence",
                claim_confidence=identity_confidence if identity_confidence is not None else 0.0,
                claim_source="local_portrait_match_uncalibrated_positive_guard",
                observation=observation,
                backend_name=backend_name,
                checked_age_s=checked_age_s,
                freshness_limit_s=freshness_limit_s,
                freshness_clock=freshness_clock,
                observation_fresh=observation_fresh,
                signals=signals,
                policy=policy,
                force_redact_identity=True,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="likely_reference_user",
            matches_reference_user=True,
            policy_recommendation="calm_personalization_only",
            block_reason=None,
            claim_confidence=identity_confidence,
            claim_source=(
                "local_portrait_match_temporal_fusion"
                if temporal_state in _STABLE_TEMPORAL_STATES
                else "local_portrait_match"
            ),
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
        )

    if canonical_state == "known_other_user":
        if identity_confidence is None or identity_confidence < policy.min_other_user_confidence:
            return _build_snapshot(
                observed_at=observed_at,
                state="ambiguous_identity",
                matches_reference_user=False,
                policy_recommendation="blocked",
                block_reason="insufficient_other_user_confidence",
                claim_confidence=identity_confidence if identity_confidence is not None else 0.0,
                claim_source="local_portrait_match_uncertainty_guard",
                observation=observation,
                backend_name=backend_name,
                checked_age_s=checked_age_s,
                freshness_limit_s=freshness_limit_s,
                freshness_clock=freshness_clock,
                observation_fresh=observation_fresh,
                signals=signals,
                policy=policy,
                force_redact_identity=True,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="known_other_user",
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason="other_enrolled_user_detected",
            claim_confidence=identity_confidence,
            claim_source=(
                "local_portrait_match_temporal_fusion"
                if temporal_state in _STABLE_TEMPORAL_STATES
                else "local_portrait_match"
            ),
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
        )

    if canonical_state == "ambiguous_identity":
        return _build_snapshot(
            observed_at=observed_at,
            state="ambiguous_identity",
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason="ambiguous_identity",
            claim_confidence=identity_confidence if identity_confidence is not None else 0.0,
            claim_source="local_portrait_match",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    if canonical_state == "uncertain_match":
        return _build_snapshot(
            observed_at=observed_at,
            state="uncertain_match",
            matches_reference_user=False,
            policy_recommendation="confirm_first",
            block_reason="portrait_match_uncertain",
            claim_confidence=identity_confidence if identity_confidence is not None else 0.0,
            claim_source="local_portrait_match",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    if canonical_state == "provider_unavailable":
        return _build_snapshot(
            observed_at=observed_at,
            state="provider_unavailable",
            matches_reference_user=False,
            policy_recommendation="unavailable",
            block_reason="provider_unavailable",
            claim_confidence=0.0,
            claim_source="local_portrait_match",
            observation=observation,
            backend_name=backend_name,
            checked_age_s=checked_age_s,
            freshness_limit_s=freshness_limit_s,
            freshness_clock=freshness_clock,
            observation_fresh=observation_fresh,
            signals=signals,
            policy=policy,
            force_redact_identity=True,
        )

    return _build_snapshot(
        observed_at=observed_at,
        state=canonical_state,
        matches_reference_user=False,
        policy_recommendation="unavailable",
        block_reason=canonical_state or "provider_unavailable",
        claim_confidence=identity_confidence if identity_confidence is not None else 0.0,
        claim_source="local_portrait_match",
        observation=observation,
        backend_name=backend_name,
        checked_age_s=checked_age_s,
        freshness_limit_s=freshness_limit_s,
        freshness_clock=freshness_clock,
        observation_fresh=observation_fresh,
        signals=signals,
        policy=policy,
        force_redact_identity=True,
    )


def _build_snapshot(
    *,
    observed_at: float | None,
    state: str,
    matches_reference_user: bool,
    policy_recommendation: str,
    block_reason: str | None,
    claim_confidence: float,
    claim_source: str,
    observation: object | None,
    backend_name: str | None,
    checked_age_s: float | None,
    freshness_limit_s: float | None,
    freshness_clock: str | None,
    observation_fresh: bool | None,
    signals: _EvidenceSignals,
    policy: PortraitMatchPolicy,
    force_redact_identity: bool = False,
) -> PortraitMatchSnapshot:
    redact_identity = force_redact_identity or _should_redact_identity(
        state=state,
        claim_confidence=claim_confidence,
        policy=policy,
    )
    matched_user_id = normalize_text(getattr(observation, "matched_user_id", None)) or None
    matched_user_display_name = (
        normalize_text(getattr(observation, "matched_user_display_name", None)) or None
    )
    if redact_identity:
        matched_user_id = None
        matched_user_display_name = None

    claim_score = _coalesce_score(claim_confidence)
    return PortraitMatchSnapshot(
        observed_at=observed_at,
        state=state,
        matches_reference_user=matches_reference_user,
        policy_recommendation=policy_recommendation,
        block_reason=block_reason,
        claim=RuntimeClaimMetadata(
            confidence=claim_score if claim_score is not None else 0.0,
            source=claim_source,
            requires_confirmation=True,
        ),
        similarity_score=coerce_optional_float(getattr(observation, "similarity_score", None)),
        checked_age_s=checked_age_s,
        live_face_count=coerce_optional_int(getattr(observation, "live_face_count", None)),
        reference_face_count=coerce_optional_int(getattr(observation, "reference_face_count", None)),
        reference_image_count=coerce_optional_int(getattr(observation, "reference_image_count", None)),
        matched_user_id=matched_user_id,
        matched_user_display_name=matched_user_display_name,
        candidate_user_count=coerce_optional_int(getattr(observation, "candidate_user_count", None)),
        temporal_state=normalize_text(getattr(observation, "temporal_state", None)) or None,
        temporal_observation_count=coerce_optional_int(
            getattr(observation, "temporal_observation_count", None)
        ),
        fused_confidence=_coalesce_score(getattr(observation, "fused_confidence", None)),
        backend_name=backend_name,
        observation_fresh=observation_fresh,
        freshness_limit_s=freshness_limit_s,
        freshness_clock=freshness_clock,
        quality_score=signals.quality_score,
        quality_state=signals.quality_state,
        liveness_confidence=signals.liveness_confidence,
        liveness_state=signals.liveness_state,
        presentation_attack_detected=signals.presentation_attack_detected,
        open_set_risk=signals.open_set_risk,
        gallery_uncertainty=signals.gallery_uncertainty,
        embedding_uncertainty=signals.embedding_uncertainty,
    )


def _extract_evidence_signals(
    *,
    facts: dict[str, object],
    observation: object | None,
) -> _EvidenceSignals:
    portrait = coerce_mapping(facts.get("portrait_match"))
    biometrics = coerce_mapping(facts.get("biometrics"))
    camera = coerce_mapping(facts.get("camera"))
    liveness = (
        coerce_mapping(portrait.get("liveness"))
        or coerce_mapping(portrait.get("pad"))
        or coerce_mapping(biometrics.get("liveness"))
        or coerce_mapping(facts.get("face_liveness"))
        or coerce_mapping(facts.get("liveness"))
        or coerce_mapping(camera.get("liveness"))
    )
    quality = (
        coerce_mapping(portrait.get("quality"))
        or coerce_mapping(biometrics.get("quality"))
        or coerce_mapping(facts.get("face_quality"))
        or coerce_mapping(facts.get("fiqa"))
        or coerce_mapping(camera.get("face_quality"))
    )
    uncertainty = (
        coerce_mapping(portrait.get("uncertainty"))
        or coerce_mapping(biometrics.get("uncertainty"))
        or coerce_mapping(facts.get("identity_uncertainty"))
        or coerce_mapping(facts.get("uncertainty"))
    )
    presentation = (
        coerce_mapping(facts.get("presentation_attack"))
        or coerce_mapping(liveness.get("presentation_attack"))
        or coerce_mapping(biometrics.get("presentation_attack"))
    )

    quality_score = _first_score(
        getattr(observation, "quality_score", None),
        getattr(observation, "face_quality_score", None),
        getattr(observation, "fiqa_score", None),
        quality.get("score"),
        quality.get("quality_score"),
        quality.get("fiqa_score"),
        portrait.get("quality_score"),
        camera.get("face_quality_score"),
        facts.get("face_quality_score"),
    )
    quality_state = _first_text(
        getattr(observation, "quality_state", None),
        getattr(observation, "face_quality_state", None),
        quality.get("state"),
        quality.get("quality_state"),
        portrait.get("quality_state"),
        camera.get("face_quality_state"),
    )

    liveness_confidence = _first_score(
        getattr(observation, "liveness_confidence", None),
        getattr(observation, "bonafide_confidence", None),
        getattr(observation, "pad_confidence", None),
        liveness.get("confidence"),
        liveness.get("liveness_confidence"),
        liveness.get("bonafide_confidence"),
        liveness.get("pad_confidence"),
        portrait.get("liveness_confidence"),
        facts.get("liveness_confidence"),
    )
    spoof_score = _first_score(
        getattr(observation, "spoof_score", None),
        getattr(observation, "presentation_attack_score", None),
        liveness.get("spoof_score"),
        liveness.get("attack_score"),
        liveness.get("presentation_attack_score"),
        presentation.get("score"),
    )
    if liveness_confidence is None and spoof_score is not None:
        liveness_confidence = round(max(0.0, 1.0 - spoof_score), 6)

    liveness_state = _canonicalize_liveness_state(
        _first_text(
            getattr(observation, "liveness_state", None),
            getattr(observation, "pad_state", None),
            getattr(observation, "presentation_attack_state", None),
            liveness.get("state"),
            liveness.get("liveness_state"),
            liveness.get("pad_state"),
            presentation.get("state"),
            facts.get("liveness_state"),
        )
    )
    presentation_attack_detected = _first_bool(
        getattr(observation, "presentation_attack_detected", None),
        getattr(observation, "spoof_detected", None),
        liveness.get("spoof_detected"),
        liveness.get("presentation_attack_detected"),
        presentation.get("detected"),
        presentation.get("is_attack"),
        facts.get("presentation_attack_detected"),
    )

    gallery_uncertainty = _first_score(
        getattr(observation, "gallery_uncertainty", None),
        uncertainty.get("gallery_uncertainty"),
        uncertainty.get("class_overlap_uncertainty"),
    )
    embedding_uncertainty = _first_score(
        getattr(observation, "embedding_uncertainty", None),
        uncertainty.get("embedding_uncertainty"),
        uncertainty.get("sample_uncertainty"),
    )
    open_set_risk = _first_score(
        getattr(observation, "open_set_risk", None),
        getattr(observation, "unknown_identity_risk", None),
        uncertainty.get("open_set_risk"),
        uncertainty.get("unknown_identity_risk"),
        portrait.get("open_set_risk"),
        facts.get("open_set_risk"),
    )
    if open_set_risk is None:
        candidates = [value for value in (gallery_uncertainty, embedding_uncertainty) if value is not None]
        if candidates:
            open_set_risk = max(candidates)

    return _EvidenceSignals(
        quality_score=quality_score,
        quality_state=quality_state,
        liveness_confidence=liveness_confidence,
        liveness_state=liveness_state,
        presentation_attack_detected=presentation_attack_detected,
        open_set_risk=open_set_risk,
        gallery_uncertainty=gallery_uncertainty,
        embedding_uncertainty=embedding_uncertainty,
    )


def _effective_identity_confidence(
    *,
    observation: object,
    signals: _EvidenceSignals,
    temporal_state: str | None,
) -> float | None:
    raw_confidence = _coalesce_score(getattr(observation, "confidence", None))
    fused_confidence = _coalesce_score(getattr(observation, "fused_confidence", None))
    if temporal_state in _STABLE_TEMPORAL_STATES and fused_confidence is not None:
        base = fused_confidence
    else:
        base = _first_conservative_score(raw_confidence, fused_confidence)
    if base is None:
        return None

    caps: list[float] = [base]
    if signals.quality_score is not None:
        caps.append(signals.quality_score)
    if signals.liveness_confidence is not None:
        caps.append(signals.liveness_confidence)
    if signals.open_set_risk is not None:
        caps.append(max(0.0, 1.0 - signals.open_set_risk))
    return round(max(0.0, min(caps)), 6)


def _freshness_limit_s(state: str, policy: PortraitMatchPolicy) -> float:
    if state == "likely_reference_user":
        return policy.max_reference_observation_age_s
    return policy.max_nonmatch_observation_age_s


def _estimate_checked_age_s(
    *,
    checked_at: object | None,
    observed_at: float | None,
    now_monotonic: float | None,
) -> tuple[float | None, str | None]:
    raw_checked_at = coerce_optional_float(checked_at)
    if raw_checked_at is None:
        return None, None

    wall_now = _coalesce_non_negative_float(observed_at)
    mono_now = _coalesce_non_negative_float(now_monotonic)
    if mono_now is None:
        mono_now = time.monotonic()

    candidates: list[tuple[float, str]] = []
    if 0.0 <= raw_checked_at <= mono_now + 60.0:
        mono_age = mono_now - raw_checked_at
        if mono_age >= -0.25:
            candidates.append((round(max(mono_age, 0.0), 3), "monotonic"))
    if wall_now is not None and raw_checked_at >= 946684800.0 and raw_checked_at <= wall_now + 60.0:
        wall_age = wall_now - raw_checked_at
        if wall_age >= -5.0:
            candidates.append((round(max(wall_age, 0.0), 3), "wall"))

    if not candidates:
        return None, None
    age_s, clock = min(candidates, key=lambda item: item[0])
    return age_s, clock


def _open_set_block_reason(
    *,
    state: str,
    signals: _EvidenceSignals,
    candidate_user_count: int | None,
    temporal_state: str | None,
    policy: PortraitMatchPolicy,
) -> str | None:
    if signals.open_set_risk is not None and signals.open_set_risk > policy.max_open_set_risk:
        return "open_set_risk_high"
    if state == "likely_reference_user" and candidate_user_count is not None and candidate_user_count > 1:
        if temporal_state not in _STABLE_TEMPORAL_STATES:
            return "multiple_identity_candidates"
    return None


def _is_low_quality(signals: _EvidenceSignals, policy: PortraitMatchPolicy) -> bool:
    if signals.quality_score is not None and signals.quality_score < policy.min_quality_score:
        return True
    return signals.quality_state in _LOW_QUALITY_STATES


def _is_presentation_attack(signals: _EvidenceSignals) -> bool:
    if signals.presentation_attack_detected is True:
        return True
    return signals.liveness_state in _LIVENESS_ATTACK_STATES


def _has_liveness_shortfall(
    *,
    signals: _EvidenceSignals,
    policy: PortraitMatchPolicy,
) -> bool:
    if signals.liveness_state in _LIVENESS_FAIL_STATES:
        return True
    if signals.liveness_confidence is None:
        return False
    if signals.liveness_state in _LIVENESS_PASS_STATES:
        return False
    return signals.liveness_confidence < policy.min_liveness_confidence


def _observation_has_no_visible_face(observation: object, canonical_state: str) -> bool:
    if canonical_state == "no_visible_person":
        return True
    live_face_count = coerce_optional_int(getattr(observation, "live_face_count", None))
    return live_face_count == 0


def _camera_person_visible(camera: dict[str, object]) -> bool | None:
    return _first_bool(
        camera.get("person_visible"),
        camera.get("person_present"),
        camera.get("face_visible"),
        camera.get("has_person"),
    )


def _should_redact_identity(
    *,
    state: str,
    claim_confidence: float | None,
    policy: PortraitMatchPolicy,
) -> bool:
    if not policy.redact_identity_when_uncertain:
        return False
    if state == "likely_reference_user" and claim_confidence is not None:
        return claim_confidence < policy.min_reference_confidence
    if state == "known_other_user" and claim_confidence is not None:
        return claim_confidence < policy.min_other_user_confidence
    return True


def _canonicalize_observation_state(value: object | None) -> str:
    state = normalize_text(value) or ""
    if not state:
        return "provider_unavailable"
    for canonical, aliases in _CANONICAL_STATE_ALIASES.items():
        if state in aliases:
            return canonical
    if "spoof" in state or "presentation_attack" in state or "replay" in state:
        return "blocked_presentation_attack"
    if "ambiguous" in state or "multiple" in state or "collision" in state:
        return "ambiguous_identity"
    if "other_user" in state or "other_enrolled" in state or ("other" in state and "user" in state):
        return "known_other_user"
    if "reference_user" in state or ("reference" in state and "match" in state) or state == "match":
        return "likely_reference_user"
    if "uncertain" in state or "low_confidence" in state or "confirm" in state:
        return "uncertain_match"
    if "no_face" in state or "no_person" in state or "no_visible" in state:
        return "no_visible_person"
    return state


def _canonicalize_liveness_state(value: str | None) -> str | None:
    state = normalize_text(value) or None
    if state is None:
        return None
    state_text = str(state)
    if state_text in _LIVENESS_ATTACK_STATES | _LIVENESS_FAIL_STATES | _LIVENESS_PASS_STATES:
        return state_text
    if "spoof" in state_text or "attack" in state_text or "replay" in state_text:
        return "presentation_attack"
    return state_text


def _first_conservative_score(*values: float | None) -> float | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return round(min(candidates), 6)


def _first_score(*values: object | None) -> float | None:
    for value in values:
        score = _coalesce_score(value)
        if score is not None:
            return score
    return None


def _first_text(*values: object | None) -> str | None:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return None


def _first_bool(*values: object | None) -> bool | None:
    for value in values:
        parsed = coerce_optional_bool(value)
        if parsed is not None:
            return parsed
    return None


def _coalesce_score(value: object | None) -> float | None:
    ratio = coerce_optional_ratio(value)
    if ratio is not None:
        return ratio
    raw = coerce_optional_float(value)
    if raw is None:
        return None
    if 0.0 <= raw <= 100.0:
        return round(raw / 100.0, 6)
    return None


def _coalesce_non_negative_float(value: object | None) -> float | None:
    raw = coerce_optional_float(value)
    if raw is None or raw < 0.0:
        return None
    return raw


def _provider_name(provider: PortraitMatchProvider) -> str:
    return normalize_text(getattr(getattr(provider, "backend", None), "name", None)) or "local_portrait_match"


__all__ = [
    "PortraitMatchPolicy",
    "PortraitMatchSnapshot",
    "derive_portrait_match",
]
