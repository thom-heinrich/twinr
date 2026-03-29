# CHANGELOG: 2026-03-29
# BUG-1: Current ambiguous-room evidence could be bypassed by a positive identity_fusion snapshot.
# BUG-2: voice_status="known_other_user" was silently treated like unknown_voice, allowing confirm-first instead of a hard block.
# BUG-3: identity_fusion snapshots were propagated with their original requires_confirmation flag, which could accidentally strengthen this intentionally weaker surface.
# SEC-1: # BREAKING: to_automation_facts() no longer exposes identity_fusion_matched_user_id by default; opt in via include_sensitive_identity_reference=True.
# SEC-2: # BREAKING: stable temporal identity fusion can still veto or block, but it no longer upgrades to a positive main-user hint on stale-only evidence.
# IMP-1: Replace naive confidence averaging with lightweight quality-aware score fusion that accounts for freshness, temporal stability, missing modalities, and live speaker association.
# IMP-2: Add explicit handling for ambiguous and uncertain voice matcher states, plus stronger open-set gating when a known other enrolled user is the best live match.

"""Derive a conservative known-user hint from live voice and room context.

This surface intentionally stays weaker than identity. It combines the current
local voice-profile result with a clear single-person room context and emits a
hint that later policy can use for calm personalization only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math

from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_optional_ratio,
    normalize_text,
)
from twinr.proactive.runtime.identity_fusion import MultimodalIdentityFusionSnapshot
from twinr.proactive.runtime.portrait_match import PortraitMatchSnapshot
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot


_DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S = 120.0
_MAX_FUTURE_SKEW_S = 5.0
_CONTEXT_QUALITY_WEIGHT = 0.35
_SPEAKER_ASSOCIATION_WEIGHT = 0.45
_TEMPORAL_IDENTITY_WEIGHT = 1.10
_ALLOWED_VOICE_STATUSES = frozenset(
    {
        "likely_user",
        "uncertain",
        "unknown_voice",
        "known_other_user",
        "uncertain_match",
        "ambiguous_match",
    }
)


def _default_claim() -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(confidence=0.0, source="voice_profile_runtime", requires_confirmation=True)


@dataclass(frozen=True, slots=True)
class KnownUserHintSnapshot:
    """Describe whether current live context likely belongs to the main user."""

    observed_at: float | None = None
    state: str = "voice_signal_unavailable"
    matches_main_user: bool = False
    policy_recommendation: str = "blocked"
    block_reason: str | None = None
    claim: RuntimeClaimMetadata = field(default_factory=_default_claim)
    voice_status: str | None = None
    voice_confidence: float | None = None
    voice_checked_age_s: float | None = None
    ambiguous_room_reason: str | None = None
    speaker_association_state: str | None = None
    portrait_match_state: str | None = None
    portrait_match_confidence: float | None = None
    portrait_match_temporal_state: str | None = None
    portrait_match_fused_confidence: float | None = None
    portrait_match_observation_count: int | None = None
    identity_fusion_state: str | None = None
    identity_fusion_confidence: float | None = None
    identity_fusion_matched_user_id: str | None = None
    identity_fusion_temporal_state: str | None = None
    identity_fusion_session_consistency_state: str | None = None
    identity_fusion_track_consistency_state: str | None = None
    identity_fusion_session_observation_count: int | None = None
    identity_fusion_track_observation_count: int | None = None

    # BREAKING: identity_fusion_matched_user_id is now redacted by default because
    # this hint surface is intentionally weaker than identity.
    def to_automation_facts(
        self,
        *,
        include_sensitive_identity_reference: bool = False,
    ) -> dict[str, object]:
        """Serialize the known-user hint into automation-friendly facts."""

        payload = {
            "observed_at": self.observed_at,
            "state": self.state,
            "matches_main_user": self.matches_main_user,
            "policy_recommendation": self.policy_recommendation,
            "block_reason": self.block_reason,
            "voice_status": self.voice_status,
            "voice_confidence": self.voice_confidence,
            "voice_checked_age_s": self.voice_checked_age_s,
            "ambiguous_room_reason": self.ambiguous_room_reason,
            "speaker_association_state": self.speaker_association_state,
            "portrait_match_state": self.portrait_match_state,
            "portrait_match_confidence": self.portrait_match_confidence,
            "portrait_match_temporal_state": self.portrait_match_temporal_state,
            "portrait_match_fused_confidence": self.portrait_match_fused_confidence,
            "portrait_match_observation_count": self.portrait_match_observation_count,
            "identity_fusion_state": self.identity_fusion_state,
            "identity_fusion_confidence": self.identity_fusion_confidence,
            "identity_fusion_temporal_state": self.identity_fusion_temporal_state,
            "identity_fusion_session_consistency_state": self.identity_fusion_session_consistency_state,
            "identity_fusion_track_consistency_state": self.identity_fusion_track_consistency_state,
            "identity_fusion_session_observation_count": self.identity_fusion_session_observation_count,
            "identity_fusion_track_observation_count": self.identity_fusion_track_observation_count,
        }
        if include_sensitive_identity_reference:
            payload["identity_fusion_matched_user_id"] = self.identity_fusion_matched_user_id
        payload.update(self.claim.to_payload())
        return payload

    def event_data(self) -> dict[str, object]:
        """Serialize the known-user hint into compact flat event fields."""

        return {
            "known_user_hint_state": self.state,
            "known_user_hint_matches_main_user": self.matches_main_user,
            "known_user_hint_confidence": self.claim.confidence,
            "known_user_hint_policy": self.policy_recommendation,
            "known_user_hint_identity_fusion_state": self.identity_fusion_state,
        }


def derive_known_user_hint(
    *,
    observed_at: float | None,
    live_facts: dict[str, object] | object,
    voice_status: object | None,
    voice_confidence: object | None,
    voice_checked_at: object | None,
    max_voice_age_s: int | float = _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S,
    ambiguous_room_guard: AmbiguousRoomGuardSnapshot | None = None,
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
    portrait_match: PortraitMatchSnapshot | None = None,
    identity_fusion: MultimodalIdentityFusionSnapshot | None = None,
    now_utc: datetime | None = None,
) -> KnownUserHintSnapshot:
    """Return one conservative known-user hint snapshot."""

    max_voice_age_s = _coerce_max_voice_age_s(max_voice_age_s)
    guard = ambiguous_room_guard or derive_ambiguous_room_guard(
        observed_at=observed_at,
        live_facts=live_facts,
    )
    normalized_status = _normalize_voice_status(voice_status)
    normalized_confidence = coerce_optional_ratio(voice_confidence)
    checked_age_s = _voice_checked_age_s(voice_checked_at, now_utc=now_utc)
    voice_is_fresh = _is_fresh_voice_check(checked_age_s, max_voice_age_s)

    portrait_positive = portrait_match is not None and portrait_match.matches_reference_user
    portrait_uncertain = portrait_match is not None and portrait_match.state == "uncertain_match"
    portrait_conflict = portrait_match is not None and portrait_match.state in {
        "unknown_face",
        "known_other_user",
        "ambiguous_identity",
    }
    portrait_known_other_user = portrait_match is not None and portrait_match.state == "known_other_user"
    portrait_evidence_confidence = _portrait_evidence_confidence(portrait_match)
    speaker_association_confidence = _speaker_association_confidence(speaker_association)

    shared = _shared_fields(
        voice_status=normalized_status or None,
        voice_confidence=normalized_confidence,
        voice_checked_age_s=checked_age_s,
        ambiguous_room_reason=guard.reason if guard.guard_active else None,
        speaker_association=speaker_association,
        portrait_match=portrait_match,
        identity_fusion=identity_fusion,
    )

    # BREAKING: current-room ambiguity is a hard veto and always overrides a stale
    # positive temporal identity_fusion snapshot.
    if guard.guard_active:
        return _build_snapshot(
            observed_at=observed_at,
            state="blocked_ambiguous_room",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason=guard.reason,
            claim=RuntimeClaimMetadata(
                confidence=_hint_confidence(
                    (guard.claim.confidence, 1.0),
                    (speaker_association_confidence, _SPEAKER_ASSOCIATION_WEIGHT),
                    default=guard.claim.confidence,
                ) or 0.0,
                source="voice_profile_plus_ambiguous_room_guard",
                requires_confirmation=True,
            ),
            shared=shared,
        )

    identity_hint = _known_user_hint_from_identity_fusion(
        observed_at=observed_at,
        identity_fusion=identity_fusion,
        allow_positive_upgrade=_allow_positive_identity_upgrade(
            voice_is_fresh=voice_is_fresh,
            portrait_positive=portrait_positive,
            speaker_association=speaker_association,
        ),
        shared=shared,
    )
    if identity_hint is not None:
        return identity_hint

    if not normalized_status:
        if portrait_positive:
            return _build_snapshot(
                observed_at=observed_at,
                state="portrait_only_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="voice_signal_unavailable",
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (portrait_evidence_confidence, _portrait_positive_weight(portrait_match)),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        default=portrait_evidence_confidence or 0.74,
                        floor=portrait_evidence_confidence or 0.0,
                    ) or 0.74,
                    source=(
                        "local_portrait_match_temporal_fusion_plus_single_visible_person_context"
                        if portrait_match is not None and portrait_match.temporal_state == "stable_match"
                        else "local_portrait_match_plus_single_visible_person_context"
                    ),
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="voice_signal_unavailable",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason="voice_signal_unavailable",
            claim=_default_claim(),
            shared=shared,
        )

    if not voice_is_fresh:
        if portrait_positive:
            return _build_snapshot(
                observed_at=observed_at,
                state="portrait_only_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="stale_voice_signal",
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (portrait_evidence_confidence, _portrait_positive_weight(portrait_match)),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        default=portrait_evidence_confidence or 0.72,
                        floor=portrait_evidence_confidence or 0.0,
                    ) or 0.72,
                    source=(
                        "local_portrait_match_temporal_fusion_plus_single_visible_person_context"
                        if portrait_match is not None and portrait_match.temporal_state == "stable_match"
                        else "local_portrait_match_plus_single_visible_person_context"
                    ),
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="stale_voice_signal",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason="stale_voice_signal",
            claim=_default_claim(),
            shared=shared,
        )

    # Strong open-set signal from vision should already veto weak or ambiguous voice.
    if portrait_known_other_user and normalized_status in {"unknown_voice", "uncertain", "uncertain_match", "ambiguous_match"}:
        return _build_snapshot(
            observed_at=observed_at,
            state="other_enrolled_user_visible",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason="known_other_user_visible",
            claim=RuntimeClaimMetadata(
                confidence=_hint_confidence(
                    (portrait_evidence_confidence, 1.0),
                    (normalized_confidence, 0.25),
                    default=portrait_evidence_confidence or 0.82,
                    floor=portrait_evidence_confidence or 0.0,
                ) or 0.82,
                source="portrait_match_known_other_user",
                requires_confirmation=True,
            ),
            shared=shared,
        )

    voice_weight = _voice_weight(normalized_status, checked_age_s, max_voice_age_s)

    # BREAKING: a fresh match to another enrolled user is a hard block now.
    if normalized_status == "known_other_user":
        return _build_snapshot(
            observed_at=observed_at,
            state="other_enrolled_user_voice",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason="known_other_user_voice",
            claim=RuntimeClaimMetadata(
                confidence=_hint_confidence(
                    (normalized_confidence or 0.88, max(voice_weight, 0.90)),
                    (speaker_association_confidence, _SPEAKER_ASSOCIATION_WEIGHT),
                    default=normalized_confidence or 0.88,
                    floor=normalized_confidence or 0.0,
                ) or 0.88,
                source="voice_profile_known_other_user",
                requires_confirmation=True,
            ),
            shared=shared,
        )

    if normalized_status == "ambiguous_match":
        if portrait_positive:
            return _build_snapshot(
                observed_at=observed_at,
                state="ambiguous_voice_multimodal_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="voice_match_ambiguous",
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (normalized_confidence or 0.50, voice_weight),
                        (portrait_evidence_confidence, _portrait_positive_weight(portrait_match)),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        default=max(normalized_confidence or 0.50, portrait_evidence_confidence or 0.66),
                    ) or 0.66,
                    source=(
                        "ambiguous_voice_plus_temporal_portrait_match_plus_context"
                        if portrait_match is not None and portrait_match.temporal_state == "stable_match"
                        else "ambiguous_voice_plus_portrait_match_plus_context"
                    ),
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="ambiguous_voice_match",
            matches_main_user=False,
            policy_recommendation="confirm_first",
            block_reason="voice_match_ambiguous",
            claim=RuntimeClaimMetadata(
                confidence=normalized_confidence or 0.50,
                source="voice_profile_runtime",
                requires_confirmation=True,
            ),
            shared=shared,
        )

    if normalized_status == "likely_user":
        if portrait_positive:
            return _build_snapshot(
                observed_at=observed_at,
                state="likely_main_user_multimodal",
                matches_main_user=True,
                policy_recommendation="calm_personalization_only",
                block_reason=None,
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (normalized_confidence or 0.72, voice_weight),
                        (portrait_evidence_confidence, _portrait_positive_weight(portrait_match)),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        (speaker_association_confidence, _SPEAKER_ASSOCIATION_WEIGHT),
                        default=max(normalized_confidence or 0.72, portrait_evidence_confidence or 0.78),
                        floor=normalized_confidence or 0.0,
                    ) or max(normalized_confidence or 0.72, portrait_evidence_confidence or 0.78),
                    source=(
                        "voice_profile_plus_temporal_portrait_match_plus_single_visible_person_context"
                        if portrait_match is not None and portrait_match.temporal_state == "stable_match"
                        else "voice_profile_plus_portrait_match_plus_single_visible_person_context"
                    ),
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        if portrait_conflict:
            return _build_snapshot(
                observed_at=observed_at,
                state="modality_conflict",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="portrait_voice_mismatch",
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (normalized_confidence or 0.72, voice_weight),
                        (portrait_evidence_confidence, 0.90),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        default=0.52,
                        floor=0.52,
                    ) or 0.52,
                    source="voice_profile_plus_portrait_mismatch",
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        if portrait_uncertain:
            return _build_snapshot(
                observed_at=observed_at,
                state="uncertain_multimodal_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="portrait_match_uncertain",
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (normalized_confidence or 0.72, voice_weight),
                        (portrait_evidence_confidence, 0.60),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        default=normalized_confidence or 0.62,
                    ) or (normalized_confidence or 0.62),
                    source="voice_profile_plus_uncertain_portrait_match",
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="likely_main_user",
            matches_main_user=True,
            policy_recommendation="calm_personalization_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=_hint_confidence(
                    (normalized_confidence or 0.72, voice_weight),
                    (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                    (speaker_association_confidence, _SPEAKER_ASSOCIATION_WEIGHT),
                    default=normalized_confidence or 0.72,
                    floor=normalized_confidence or 0.0,
                ) or (normalized_confidence or 0.72),
                source="voice_profile_plus_single_visible_person_context",
                requires_confirmation=True,
            ),
            shared=shared,
        )

    if normalized_status in {"uncertain", "uncertain_match"}:
        if portrait_positive:
            return _build_snapshot(
                observed_at=observed_at,
                state="uncertain_voice_multimodal_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="voice_signal_uncertain",
                claim=RuntimeClaimMetadata(
                    confidence=_hint_confidence(
                        (normalized_confidence or 0.58, voice_weight),
                        (portrait_evidence_confidence, _portrait_positive_weight(portrait_match)),
                        (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                        default=portrait_evidence_confidence or 0.66,
                        floor=normalized_confidence or 0.0,
                    ) or (portrait_evidence_confidence or 0.66),
                    source=(
                        "uncertain_voice_plus_temporal_portrait_match_plus_context"
                        if portrait_match is not None and portrait_match.temporal_state == "stable_match"
                        else "uncertain_voice_plus_portrait_match_plus_context"
                    ),
                    requires_confirmation=True,
                ),
                shared=shared,
            )
        return _build_snapshot(
            observed_at=observed_at,
            state="uncertain_voice_match",
            matches_main_user=False,
            policy_recommendation="confirm_first",
            block_reason="voice_signal_uncertain",
            claim=RuntimeClaimMetadata(
                confidence=normalized_confidence or 0.58,
                source="voice_profile_runtime",
                requires_confirmation=True,
            ),
            shared=shared,
        )

    if portrait_positive:
        return _build_snapshot(
            observed_at=observed_at,
            state="unknown_voice_multimodal_match",
            matches_main_user=False,
            policy_recommendation="confirm_first",
            block_reason="unknown_voice",
            claim=RuntimeClaimMetadata(
                confidence=_hint_confidence(
                    (normalized_confidence or 0.36, voice_weight),
                    (portrait_evidence_confidence, _portrait_positive_weight(portrait_match)),
                    (guard.claim.confidence, _CONTEXT_QUALITY_WEIGHT),
                    default=portrait_evidence_confidence or 0.62,
                    floor=portrait_evidence_confidence or 0.0,
                ) or (portrait_evidence_confidence or 0.62),
                source=(
                    "unknown_voice_plus_temporal_portrait_match_plus_context"
                    if portrait_match is not None and portrait_match.temporal_state == "stable_match"
                    else "unknown_voice_plus_portrait_match_plus_context"
                ),
                requires_confirmation=True,
            ),
            shared=shared,
        )

    return _build_snapshot(
        observed_at=observed_at,
        state="unknown_voice",
        matches_main_user=False,
        policy_recommendation="confirm_first",
        block_reason="unknown_voice",
        claim=RuntimeClaimMetadata(
            confidence=normalized_confidence or 0.36,
            source="voice_profile_runtime",
            requires_confirmation=True,
        ),
        shared=shared,
    )


def _normalize_voice_status(value: object | None) -> str:
    """Return one normalized voice-status token or an empty string."""

    normalized = normalize_text(value).lower()
    if normalized in _ALLOWED_VOICE_STATUSES:
        return normalized
    return ""


def _known_user_hint_from_identity_fusion(
    *,
    observed_at: float | None,
    identity_fusion: MultimodalIdentityFusionSnapshot | None,
    allow_positive_upgrade: bool,
    shared: dict[str, object],
) -> KnownUserHintSnapshot | None:
    """Upgrade the hint when temporal multimodal fusion is stronger than point-in-time evidence."""

    if identity_fusion is None:
        return None

    claim = _force_confirmation_claim(identity_fusion.claim)

    if identity_fusion.state == "blocked_ambiguous_room":
        return _build_snapshot(
            observed_at=observed_at,
            state="blocked_ambiguous_room",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason=identity_fusion.block_reason or shared.get("ambiguous_room_reason"),
            claim=claim,
            shared=shared,
        )

    if identity_fusion.state == "modality_conflict":
        return _build_snapshot(
            observed_at=observed_at,
            state="modality_conflict",
            matches_main_user=False,
            policy_recommendation="confirm_first",
            block_reason=identity_fusion.block_reason or "identity_modality_conflict",
            claim=claim,
            shared=shared,
        )

    if identity_fusion.state in {"stable_other_enrolled_user", "other_enrolled_user_candidate"}:
        return _build_snapshot(
            observed_at=observed_at,
            state="other_enrolled_user_visible",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason=identity_fusion.block_reason or "other_enrolled_user_detected",
            claim=claim,
            shared=shared,
        )

    if identity_fusion.state == "stable_main_user_multimodal" and allow_positive_upgrade:
        return _build_snapshot(
            observed_at=observed_at,
            state="likely_main_user_temporal_multimodal",
            matches_main_user=True,
            policy_recommendation="calm_personalization_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=_hint_confidence(
                    (identity_fusion.claim.confidence, _TEMPORAL_IDENTITY_WEIGHT),
                    (shared.get("voice_confidence"), 0.60 if shared.get("voice_status") == "likely_user" else 0.0),
                    (
                        shared.get("portrait_match_fused_confidence"),
                        0.90 if shared.get("portrait_match_temporal_state") == "stable_match" else 0.0,
                    ),
                    (
                        shared.get("portrait_match_confidence"),
                        0.70 if shared.get("portrait_match_state") else 0.0,
                    ),
                    default=identity_fusion.claim.confidence,
                    floor=identity_fusion.claim.confidence,
                    ceiling=0.995,
                ) or identity_fusion.claim.confidence,
                source=identity_fusion.claim.source,
                requires_confirmation=True,
            ),
            shared=shared,
        )

    return None


def _shared_fields(
    *,
    voice_status: str | None,
    voice_confidence: float | None,
    voice_checked_age_s: float | None,
    ambiguous_room_reason: str | None,
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
    portrait_match: PortraitMatchSnapshot | None,
    identity_fusion: MultimodalIdentityFusionSnapshot | None,
) -> dict[str, object]:
    speaker_association_state = None if speaker_association is None else speaker_association.state
    portrait_match_state = None if portrait_match is None else portrait_match.state
    portrait_match_confidence = None if portrait_match is None else portrait_match.claim.confidence
    portrait_match_temporal_state = None if portrait_match is None else portrait_match.temporal_state
    portrait_match_fused_confidence = None if portrait_match is None else portrait_match.fused_confidence
    portrait_match_observation_count = None if portrait_match is None else portrait_match.temporal_observation_count

    return {
        "voice_status": voice_status,
        "voice_confidence": voice_confidence,
        "voice_checked_age_s": voice_checked_age_s,
        "ambiguous_room_reason": ambiguous_room_reason,
        "speaker_association_state": speaker_association_state,
        "portrait_match_state": portrait_match_state,
        "portrait_match_confidence": portrait_match_confidence,
        "portrait_match_temporal_state": portrait_match_temporal_state,
        "portrait_match_fused_confidence": portrait_match_fused_confidence,
        "portrait_match_observation_count": portrait_match_observation_count,
        "identity_fusion_state": None if identity_fusion is None else identity_fusion.state,
        "identity_fusion_confidence": None if identity_fusion is None else identity_fusion.claim.confidence,
        "identity_fusion_matched_user_id": None if identity_fusion is None else identity_fusion.matched_user_id,
        "identity_fusion_temporal_state": None if identity_fusion is None else identity_fusion.temporal_state,
        "identity_fusion_session_consistency_state": (
            None if identity_fusion is None else identity_fusion.session_consistency_state
        ),
        "identity_fusion_track_consistency_state": (
            None if identity_fusion is None else identity_fusion.track_consistency_state
        ),
        "identity_fusion_session_observation_count": (
            None if identity_fusion is None else identity_fusion.session_observation_count
        ),
        "identity_fusion_track_observation_count": (
            None if identity_fusion is None else identity_fusion.track_observation_count
        ),
    }


def _build_snapshot(
    *,
    observed_at: float | None,
    state: str,
    matches_main_user: bool,
    policy_recommendation: str,
    block_reason: str | None,
    claim: RuntimeClaimMetadata,
    shared: dict[str, object],
) -> KnownUserHintSnapshot:
    return KnownUserHintSnapshot(
        observed_at=observed_at,
        state=state,
        matches_main_user=matches_main_user,
        policy_recommendation=policy_recommendation,
        block_reason=block_reason,
        claim=_force_confirmation_claim(claim),
        **shared,
    )


def _force_confirmation_claim(claim: RuntimeClaimMetadata) -> RuntimeClaimMetadata:
    return RuntimeClaimMetadata(
        confidence=coerce_optional_ratio(claim.confidence) or 0.0,
        source=claim.source,
        requires_confirmation=True,
    )


def _allow_positive_identity_upgrade(
    *,
    voice_is_fresh: bool,
    portrait_positive: bool,
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
) -> bool:
    if voice_is_fresh:
        return True
    if portrait_positive:
        return True
    if speaker_association is not None and speaker_association.associated:
        return True
    return False


def _hint_confidence(
    *parts: tuple[float | None, float],
    default: float | None = None,
    floor: float | None = None,
    ceiling: float | None = None,
) -> float | None:
    weighted_total = 0.0
    weight_total = 0.0

    for score, weight in parts:
        normalized_score = coerce_optional_ratio(score)
        normalized_weight = max(0.0, float(weight))
        if normalized_score is None or normalized_weight <= 0.0:
            continue
        weighted_total += normalized_score * normalized_weight
        weight_total += normalized_weight

    if weight_total <= 0.0:
        fused = default
    else:
        fused = weighted_total / weight_total

    fused = coerce_optional_ratio(fused)
    if fused is None:
        return None

    if floor is not None:
        fused = max(fused, coerce_optional_ratio(floor) or 0.0)
    if ceiling is not None:
        fused = min(fused, coerce_optional_ratio(ceiling) or 1.0)

    return round(fused, 3)


def _coerce_max_voice_age_s(value: int | float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S

    if math.isnan(parsed) or math.isinf(parsed):
        return _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S

    return max(1.0, parsed)


def _is_fresh_voice_check(age_s: float | None, max_voice_age_s: float) -> bool:
    return age_s is not None and age_s <= max_voice_age_s


def _voice_weight(status: str, checked_age_s: float | None, max_voice_age_s: float) -> float:
    if checked_age_s is None or checked_age_s > max_voice_age_s:
        return 0.0

    freshness = 1.0 - min(max(checked_age_s, 0.0) / max_voice_age_s, 1.0)
    freshness_multiplier = 0.55 + (0.45 * freshness)
    base = {
        "likely_user": 1.00,
        "known_other_user": 0.95,
        "uncertain": 0.72,
        "uncertain_match": 0.68,
        "ambiguous_match": 0.52,
        "unknown_voice": 0.40,
    }.get(status, 0.40)

    return round(base * freshness_multiplier, 3)


def _portrait_positive_weight(portrait_match: PortraitMatchSnapshot | None) -> float:
    if portrait_match is None or not portrait_match.matches_reference_user:
        return 0.0
    if portrait_match.temporal_state == "stable_match":
        return 1.0
    return 0.82


def _portrait_evidence_confidence(portrait_match: PortraitMatchSnapshot | None) -> float | None:
    if portrait_match is None:
        return None
    if portrait_match.temporal_state == "stable_match" and portrait_match.fused_confidence is not None:
        return portrait_match.fused_confidence
    return portrait_match.claim.confidence


def _speaker_association_confidence(
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
) -> float | None:
    if speaker_association is None or not speaker_association.associated:
        return None
    return coerce_optional_ratio(speaker_association.confidence)


def _voice_checked_age_s(
    value: object | None,
    *,
    now_utc: datetime | None,
) -> float | None:
    """Return the age of one voice-check timestamp in seconds."""

    checked_at = _parse_aware_utc_datetime(value)
    if checked_at is None:
        return None

    now = now_utc or datetime.now(timezone.utc)
    if checked_at > now + timedelta(seconds=_MAX_FUTURE_SKEW_S):
        return None

    return round(max(0.0, (now - checked_at).total_seconds()), 3)


def _parse_aware_utc_datetime(value: object | None) -> datetime | None:
    """Parse one optional UTC timestamp into an aware datetime."""

    if value is None:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = normalize_text(value)
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None

    if parsed.tzinfo is None:
        return None

    return parsed.astimezone(timezone.utc)


__all__ = [
    "KnownUserHintSnapshot",
    "derive_known_user_hint",
]