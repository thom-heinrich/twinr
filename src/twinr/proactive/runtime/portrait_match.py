"""Project local portrait-match observations into conservative runtime claims."""

from __future__ import annotations

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


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="local_portrait_match",
        requires_confirmation=True,
    )


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
            fused_confidence=coerce_optional_ratio(payload.get("fused_confidence")),
            backend_name=normalize_text(payload.get("backend_name")) or None,
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
    if coerce_optional_bool(camera.get("person_visible")) is not True:
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
    checked_age_s = None
    if observation.checked_at is not None and now_monotonic is not None:
        checked_age_s = round(max(0.0, now_monotonic - observation.checked_at), 3)
    if observation.state == "likely_reference_user":
        confidence = (
            coerce_optional_ratio(observation.fused_confidence)
            if normalize_text(getattr(observation, "temporal_state", None)) == "stable_match"
            else coerce_optional_ratio(observation.confidence)
        ) or 0.78
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state=observation.state,
            matches_reference_user=True,
            policy_recommendation="calm_personalization_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source=(
                    "local_portrait_match_temporal_fusion"
                    if normalize_text(getattr(observation, "temporal_state", None)) == "stable_match"
                    else "local_portrait_match"
                ),
                requires_confirmation=True,
            ),
            similarity_score=observation.similarity_score,
            checked_age_s=checked_age_s,
            live_face_count=observation.live_face_count,
            reference_face_count=observation.reference_face_count,
            reference_image_count=coerce_optional_int(getattr(observation, "reference_image_count", None)),
            matched_user_id=normalize_text(getattr(observation, "matched_user_id", None)) or None,
            matched_user_display_name=normalize_text(getattr(observation, "matched_user_display_name", None)) or None,
            candidate_user_count=coerce_optional_int(getattr(observation, "candidate_user_count", None)),
            temporal_state=normalize_text(getattr(observation, "temporal_state", None)) or None,
            temporal_observation_count=coerce_optional_int(
                getattr(observation, "temporal_observation_count", None)
            ),
            fused_confidence=coerce_optional_ratio(getattr(observation, "fused_confidence", None)),
            backend_name=normalize_text(observation.backend_name) or None,
        )
    if observation.state == "known_other_user":
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state=observation.state,
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason="other_enrolled_user_detected",
            claim=RuntimeClaimMetadata(
                confidence=coerce_optional_ratio(observation.fused_confidence)
                or coerce_optional_ratio(observation.confidence)
                or 0.74,
                source="local_portrait_match_temporal_fusion",
                requires_confirmation=True,
            ),
            similarity_score=observation.similarity_score,
            checked_age_s=checked_age_s,
            live_face_count=observation.live_face_count,
            reference_face_count=observation.reference_face_count,
            reference_image_count=coerce_optional_int(getattr(observation, "reference_image_count", None)),
            matched_user_id=normalize_text(getattr(observation, "matched_user_id", None)) or None,
            matched_user_display_name=normalize_text(getattr(observation, "matched_user_display_name", None)) or None,
            candidate_user_count=coerce_optional_int(getattr(observation, "candidate_user_count", None)),
            temporal_state=normalize_text(getattr(observation, "temporal_state", None)) or None,
            temporal_observation_count=coerce_optional_int(
                getattr(observation, "temporal_observation_count", None)
            ),
            fused_confidence=coerce_optional_ratio(getattr(observation, "fused_confidence", None)),
            backend_name=normalize_text(observation.backend_name) or None,
        )
    if observation.state == "ambiguous_identity":
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state=observation.state,
            matches_reference_user=False,
            policy_recommendation="blocked",
            block_reason="ambiguous_identity",
            claim=RuntimeClaimMetadata(
                confidence=coerce_optional_ratio(observation.confidence) or 0.68,
                source="local_portrait_match",
                requires_confirmation=True,
            ),
            similarity_score=observation.similarity_score,
            checked_age_s=checked_age_s,
            live_face_count=observation.live_face_count,
            reference_face_count=observation.reference_face_count,
            reference_image_count=coerce_optional_int(getattr(observation, "reference_image_count", None)),
            matched_user_id=normalize_text(getattr(observation, "matched_user_id", None)) or None,
            matched_user_display_name=normalize_text(getattr(observation, "matched_user_display_name", None)) or None,
            candidate_user_count=coerce_optional_int(getattr(observation, "candidate_user_count", None)),
            temporal_state=normalize_text(getattr(observation, "temporal_state", None)) or None,
            temporal_observation_count=coerce_optional_int(
                getattr(observation, "temporal_observation_count", None)
            ),
            fused_confidence=coerce_optional_ratio(getattr(observation, "fused_confidence", None)),
            backend_name=normalize_text(observation.backend_name) or None,
        )
    if observation.state == "uncertain_match":
        return PortraitMatchSnapshot(
            observed_at=observed_at,
            state=observation.state,
            matches_reference_user=False,
            policy_recommendation="confirm_first",
            block_reason="portrait_match_uncertain",
            claim=RuntimeClaimMetadata(
                confidence=coerce_optional_ratio(observation.confidence) or 0.58,
                source="local_portrait_match",
                requires_confirmation=True,
            ),
            similarity_score=observation.similarity_score,
            checked_age_s=checked_age_s,
            live_face_count=observation.live_face_count,
            reference_face_count=observation.reference_face_count,
            reference_image_count=coerce_optional_int(getattr(observation, "reference_image_count", None)),
            matched_user_id=normalize_text(getattr(observation, "matched_user_id", None)) or None,
            matched_user_display_name=normalize_text(getattr(observation, "matched_user_display_name", None)) or None,
            candidate_user_count=coerce_optional_int(getattr(observation, "candidate_user_count", None)),
            temporal_state=normalize_text(getattr(observation, "temporal_state", None)) or None,
            temporal_observation_count=coerce_optional_int(
                getattr(observation, "temporal_observation_count", None)
            ),
            fused_confidence=coerce_optional_ratio(getattr(observation, "fused_confidence", None)),
            backend_name=normalize_text(observation.backend_name) or None,
        )
    return PortraitMatchSnapshot(
        observed_at=observed_at,
        state=normalize_text(observation.state) or "provider_unavailable",
        matches_reference_user=False,
        policy_recommendation="unavailable",
        block_reason=normalize_text(observation.state) or "provider_unavailable",
        claim=RuntimeClaimMetadata(
            confidence=coerce_optional_ratio(observation.confidence) or 0.0,
            source="local_portrait_match",
            requires_confirmation=True,
        ),
        similarity_score=observation.similarity_score,
        checked_age_s=checked_age_s,
        live_face_count=observation.live_face_count,
        reference_face_count=observation.reference_face_count,
        reference_image_count=coerce_optional_int(getattr(observation, "reference_image_count", None)),
        matched_user_id=normalize_text(getattr(observation, "matched_user_id", None)) or None,
        matched_user_display_name=normalize_text(getattr(observation, "matched_user_display_name", None)) or None,
        candidate_user_count=coerce_optional_int(getattr(observation, "candidate_user_count", None)),
        temporal_state=normalize_text(getattr(observation, "temporal_state", None)) or None,
        temporal_observation_count=coerce_optional_int(
            getattr(observation, "temporal_observation_count", None)
        ),
        fused_confidence=coerce_optional_ratio(getattr(observation, "fused_confidence", None)),
        backend_name=normalize_text(observation.backend_name) or None,
    )


def _provider_name(provider: PortraitMatchProvider) -> str:
    return normalize_text(getattr(getattr(provider, "backend", None), "name", None)) or "local_portrait_match"


__all__ = [
    "PortraitMatchSnapshot",
    "derive_portrait_match",
]
