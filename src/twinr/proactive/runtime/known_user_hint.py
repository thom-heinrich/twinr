"""Derive a conservative known-user hint from live voice and room context.

This surface intentionally stays weaker than identity. It combines the current
local voice-profile result with a clear single-person room context and emits a
hint that later policy can use for calm personalization only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)
from twinr.proactive.runtime.identity_fusion import MultimodalIdentityFusionSnapshot
from twinr.proactive.runtime.portrait_match import PortraitMatchSnapshot
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot


_DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S = 120
_MAX_FUTURE_SKEW_S = 5.0
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

    def to_automation_facts(self) -> dict[str, object]:
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
            "identity_fusion_matched_user_id": self.identity_fusion_matched_user_id,
            "identity_fusion_temporal_state": self.identity_fusion_temporal_state,
            "identity_fusion_session_consistency_state": self.identity_fusion_session_consistency_state,
            "identity_fusion_track_consistency_state": self.identity_fusion_track_consistency_state,
            "identity_fusion_session_observation_count": self.identity_fusion_session_observation_count,
            "identity_fusion_track_observation_count": self.identity_fusion_track_observation_count,
        }
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
    max_voice_age_s: int = _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S,
    ambiguous_room_guard: AmbiguousRoomGuardSnapshot | None = None,
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
    portrait_match: PortraitMatchSnapshot | None = None,
    identity_fusion: MultimodalIdentityFusionSnapshot | None = None,
    now_utc: datetime | None = None,
) -> KnownUserHintSnapshot:
    """Return one conservative known-user hint snapshot."""

    guard = ambiguous_room_guard or derive_ambiguous_room_guard(
        observed_at=observed_at,
        live_facts=live_facts,
    )
    normalized_status = _normalize_voice_status(voice_status)
    normalized_confidence = coerce_optional_ratio(voice_confidence)
    checked_age_s = _voice_checked_age_s(voice_checked_at, now_utc=now_utc)
    speaker_association_state = None if speaker_association is None else speaker_association.state
    portrait_match_state = None if portrait_match is None else portrait_match.state
    portrait_match_confidence = None if portrait_match is None else portrait_match.claim.confidence
    portrait_match_temporal_state = None if portrait_match is None else portrait_match.temporal_state
    portrait_match_fused_confidence = None if portrait_match is None else portrait_match.fused_confidence
    portrait_match_observation_count = None if portrait_match is None else portrait_match.temporal_observation_count
    identity_hint = _known_user_hint_from_identity_fusion(
        observed_at=observed_at,
        identity_fusion=identity_fusion,
        voice_status=normalized_status or None,
        voice_confidence=normalized_confidence,
        voice_checked_age_s=checked_age_s,
        ambiguous_room_reason=guard.reason,
        speaker_association_state=speaker_association_state,
        portrait_match_state=portrait_match_state,
        portrait_match_confidence=portrait_match_confidence,
        portrait_match_temporal_state=portrait_match_temporal_state,
        portrait_match_fused_confidence=portrait_match_fused_confidence,
        portrait_match_observation_count=portrait_match_observation_count,
    )
    if identity_hint is not None:
        return identity_hint
    portrait_positive = portrait_match is not None and portrait_match.matches_reference_user
    portrait_uncertain = portrait_match is not None and portrait_match.state == "uncertain_match"
    portrait_conflict = portrait_match is not None and portrait_match.state in {
        "unknown_face",
        "known_other_user",
        "ambiguous_identity",
    }
    portrait_evidence_confidence = portrait_match_confidence
    if portrait_match_temporal_state == "stable_match" and portrait_match_fused_confidence is not None:
        portrait_evidence_confidence = portrait_match_fused_confidence

    if guard.guard_active:
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="blocked_ambiguous_room",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason=guard.reason,
            claim=RuntimeClaimMetadata(
                confidence=guard.claim.confidence,
                source="voice_profile_plus_ambiguous_room_guard",
                requires_confirmation=True,
            ),
            voice_status=normalized_status or None,
            voice_confidence=normalized_confidence,
            voice_checked_age_s=checked_age_s,
            ambiguous_room_reason=guard.reason,
            speaker_association_state=speaker_association_state,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )
    if not normalized_status:
        if portrait_positive:
            confidence = mean_confidence((portrait_evidence_confidence, guard.claim.confidence)) or (
                portrait_evidence_confidence or 0.74
            )
            return KnownUserHintSnapshot(
                observed_at=observed_at,
                state="portrait_only_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="voice_signal_unavailable",
                claim=RuntimeClaimMetadata(
                    confidence=confidence,
                    source=(
                        "local_portrait_match_temporal_fusion_plus_single_visible_person_context"
                        if portrait_match_temporal_state == "stable_match"
                        else "local_portrait_match_plus_single_visible_person_context"
                    ),
                    requires_confirmation=True,
                ),
                voice_checked_age_s=checked_age_s,
                speaker_association_state=speaker_association_state,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="voice_signal_unavailable",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason="voice_signal_unavailable",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="voice_profile_runtime",
                requires_confirmation=True,
            ),
            voice_checked_age_s=checked_age_s,
            speaker_association_state=speaker_association_state,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )
    if checked_age_s is None or checked_age_s > max(1, int(max_voice_age_s)):
        if portrait_positive:
            confidence = mean_confidence((portrait_evidence_confidence, guard.claim.confidence)) or (
                portrait_evidence_confidence or 0.72
            )
            return KnownUserHintSnapshot(
                observed_at=observed_at,
                state="portrait_only_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="stale_voice_signal",
                claim=RuntimeClaimMetadata(
                    confidence=confidence,
                    source=(
                        "local_portrait_match_temporal_fusion_plus_single_visible_person_context"
                        if portrait_match_temporal_state == "stable_match"
                        else "local_portrait_match_plus_single_visible_person_context"
                    ),
                    requires_confirmation=True,
                ),
                voice_status=normalized_status,
                voice_confidence=normalized_confidence,
                voice_checked_age_s=checked_age_s,
                speaker_association_state=speaker_association_state,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="stale_voice_signal",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason="stale_voice_signal",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="voice_profile_runtime",
                requires_confirmation=True,
            ),
            voice_status=normalized_status,
            voice_confidence=normalized_confidence,
            voice_checked_age_s=checked_age_s,
            speaker_association_state=speaker_association_state,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )
    if normalized_status == "likely_user":
        if portrait_positive:
            confidence = mean_confidence(
                (
                    normalized_confidence or 0.72,
                    portrait_evidence_confidence,
                    guard.claim.confidence,
                    (
                        None
                        if speaker_association is None or not speaker_association.associated
                        else speaker_association.confidence
                    ),
                )
            ) or max(normalized_confidence or 0.72, portrait_evidence_confidence or 0.78)
            return KnownUserHintSnapshot(
                observed_at=observed_at,
                state="likely_main_user_multimodal",
                matches_main_user=True,
                policy_recommendation="calm_personalization_only",
                block_reason=None,
                claim=RuntimeClaimMetadata(
                    confidence=confidence,
                    source=(
                        "voice_profile_plus_temporal_portrait_match_plus_single_visible_person_context"
                        if portrait_match_temporal_state == "stable_match"
                        else "voice_profile_plus_portrait_match_plus_single_visible_person_context"
                    ),
                    requires_confirmation=True,
                ),
                voice_status=normalized_status,
                voice_confidence=normalized_confidence,
                voice_checked_age_s=checked_age_s,
                ambiguous_room_reason=None,
                speaker_association_state=speaker_association_state,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )
        if portrait_conflict:
            confidence = mean_confidence(
                (
                    normalized_confidence or 0.72,
                    portrait_evidence_confidence,
                    guard.claim.confidence,
                )
            ) or 0.52
            return KnownUserHintSnapshot(
                observed_at=observed_at,
                state="modality_conflict",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="portrait_voice_mismatch",
                claim=RuntimeClaimMetadata(
                    confidence=confidence,
                    source="voice_profile_plus_portrait_mismatch",
                    requires_confirmation=True,
                ),
                voice_status=normalized_status,
                voice_confidence=normalized_confidence,
                voice_checked_age_s=checked_age_s,
                speaker_association_state=speaker_association_state,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )
        if portrait_uncertain:
            confidence = mean_confidence(
                (
                    normalized_confidence or 0.72,
                    portrait_evidence_confidence,
                    guard.claim.confidence,
                )
            ) or (normalized_confidence or 0.62)
            return KnownUserHintSnapshot(
                observed_at=observed_at,
                state="uncertain_multimodal_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="portrait_match_uncertain",
                claim=RuntimeClaimMetadata(
                    confidence=confidence,
                    source="voice_profile_plus_uncertain_portrait_match",
                    requires_confirmation=True,
                ),
                voice_status=normalized_status,
                voice_confidence=normalized_confidence,
                voice_checked_age_s=checked_age_s,
                speaker_association_state=speaker_association_state,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )
        confidence = mean_confidence(
            (
                normalized_confidence or 0.72,
                guard.claim.confidence,
                (
                    None
                    if speaker_association is None or not speaker_association.associated
                    else speaker_association.confidence
                ),
            )
        ) or (normalized_confidence or 0.72)
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="likely_main_user",
            matches_main_user=True,
            policy_recommendation="calm_personalization_only",
            block_reason=None,
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source="voice_profile_plus_single_visible_person_context",
                requires_confirmation=True,
            ),
            voice_status=normalized_status,
            voice_confidence=normalized_confidence,
            voice_checked_age_s=checked_age_s,
            ambiguous_room_reason=None,
            speaker_association_state=speaker_association_state,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )
    if normalized_status == "uncertain":
        if portrait_positive:
            confidence = mean_confidence(
                (
                    normalized_confidence or 0.58,
                    portrait_evidence_confidence,
                    guard.claim.confidence,
                )
            ) or (portrait_evidence_confidence or 0.66)
            return KnownUserHintSnapshot(
                observed_at=observed_at,
                state="uncertain_voice_multimodal_match",
                matches_main_user=False,
                policy_recommendation="confirm_first",
                block_reason="voice_signal_uncertain",
                claim=RuntimeClaimMetadata(
                    confidence=confidence,
                    source=(
                        "uncertain_voice_plus_temporal_portrait_match_plus_context"
                        if portrait_match_temporal_state == "stable_match"
                        else "uncertain_voice_plus_portrait_match_plus_context"
                    ),
                    requires_confirmation=True,
                ),
                voice_status=normalized_status,
                voice_confidence=normalized_confidence,
                voice_checked_age_s=checked_age_s,
                speaker_association_state=speaker_association_state,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )
        return KnownUserHintSnapshot(
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
            voice_status=normalized_status,
            voice_confidence=normalized_confidence,
            voice_checked_age_s=checked_age_s,
            speaker_association_state=speaker_association_state,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )
    if portrait_positive:
        confidence = mean_confidence(
            (
                normalized_confidence or 0.36,
                portrait_evidence_confidence,
                guard.claim.confidence,
            )
        ) or (portrait_evidence_confidence or 0.62)
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="unknown_voice_multimodal_match",
            matches_main_user=False,
            policy_recommendation="confirm_first",
            block_reason="unknown_voice",
            claim=RuntimeClaimMetadata(
                confidence=confidence,
                source=(
                    "unknown_voice_plus_temporal_portrait_match_plus_context"
                    if portrait_match_temporal_state == "stable_match"
                    else "unknown_voice_plus_portrait_match_plus_context"
                ),
                requires_confirmation=True,
            ),
            voice_status=normalized_status,
            voice_confidence=normalized_confidence,
            voice_checked_age_s=checked_age_s,
            speaker_association_state=speaker_association_state,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )
    return KnownUserHintSnapshot(
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
        voice_status=normalized_status,
        voice_confidence=normalized_confidence,
        voice_checked_age_s=checked_age_s,
        speaker_association_state=speaker_association_state,
        portrait_match_state=portrait_match_state,
        portrait_match_confidence=portrait_match_confidence,
        portrait_match_temporal_state=portrait_match_temporal_state,
        portrait_match_fused_confidence=portrait_match_fused_confidence,
        portrait_match_observation_count=portrait_match_observation_count,
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
    voice_status: str | None,
    voice_confidence: float | None,
    voice_checked_age_s: float | None,
    ambiguous_room_reason: str | None,
    speaker_association_state: str | None,
    portrait_match_state: str | None,
    portrait_match_confidence: float | None,
    portrait_match_temporal_state: str | None,
    portrait_match_fused_confidence: float | None,
    portrait_match_observation_count: int | None,
) -> KnownUserHintSnapshot | None:
    """Upgrade the hint when temporal multimodal fusion is stronger than point-in-time evidence."""

    if identity_fusion is None:
        return None
    shared = {
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
        "identity_fusion_state": identity_fusion.state,
        "identity_fusion_confidence": identity_fusion.claim.confidence,
        "identity_fusion_matched_user_id": identity_fusion.matched_user_id,
        "identity_fusion_temporal_state": identity_fusion.temporal_state,
        "identity_fusion_session_consistency_state": identity_fusion.session_consistency_state,
        "identity_fusion_track_consistency_state": identity_fusion.track_consistency_state,
        "identity_fusion_session_observation_count": identity_fusion.session_observation_count,
        "identity_fusion_track_observation_count": identity_fusion.track_observation_count,
    }
    if identity_fusion.state == "blocked_ambiguous_room":
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="blocked_ambiguous_room",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason=identity_fusion.block_reason or ambiguous_room_reason,
            claim=identity_fusion.claim,
            **shared,
        )
    if identity_fusion.state == "stable_main_user_multimodal":
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="likely_main_user_temporal_multimodal",
            matches_main_user=True,
            policy_recommendation="calm_personalization_only",
            block_reason=None,
            claim=identity_fusion.claim,
            **shared,
        )
    if identity_fusion.state == "modality_conflict":
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="modality_conflict",
            matches_main_user=False,
            policy_recommendation="confirm_first",
            block_reason=identity_fusion.block_reason or "identity_modality_conflict",
            claim=identity_fusion.claim,
            **shared,
        )
    if identity_fusion.state in {"stable_other_enrolled_user", "other_enrolled_user_candidate"}:
        return KnownUserHintSnapshot(
            observed_at=observed_at,
            state="other_enrolled_user_visible",
            matches_main_user=False,
            policy_recommendation="blocked",
            block_reason=identity_fusion.block_reason or "other_enrolled_user_detected",
            claim=identity_fusion.claim,
            **shared,
        )
    return None


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
