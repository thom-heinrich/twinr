"""Fuse live identity hints with bounded temporal and session memory.

This module keeps the stateful, short-lived identity evidence tracker out of
``service.py`` and ``known_user_hint.py``. It combines conservative voice
verification, local portrait matching, speaker association, visual-anchor
history, and presence-session scoped memory into one confirm-first runtime
surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_optional_int,
    coerce_optional_ratio,
    mean_confidence,
    normalize_text,
)
from twinr.proactive.runtime.portrait_match import PortraitMatchSnapshot
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot


_DEFAULT_TEMPORAL_WINDOW_S = 180.0
_DEFAULT_MAX_OBSERVATIONS = 24
_DEFAULT_MIN_SESSION_OBSERVATIONS = 3
_DEFAULT_MIN_TRACK_OBSERVATIONS = 2
_DEFAULT_MIN_SUPPORT_RATIO = 0.67
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
    return RuntimeClaimMetadata(
        confidence=0.0,
        source="multimodal_identity_fusion",
        requires_confirmation=True,
    )


@dataclass(frozen=True, slots=True)
class MultimodalIdentityFusionConfig:
    """Store bounded tuning values for temporal identity fusion."""

    primary_user_id: str = "main_user"
    temporal_window_s: float = _DEFAULT_TEMPORAL_WINDOW_S
    max_observations: int = _DEFAULT_MAX_OBSERVATIONS
    min_session_observations: int = _DEFAULT_MIN_SESSION_OBSERVATIONS
    min_track_observations: int = _DEFAULT_MIN_TRACK_OBSERVATIONS
    min_support_ratio: float = _DEFAULT_MIN_SUPPORT_RATIO

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "MultimodalIdentityFusionConfig":
        """Derive one bounded fusion config from the global Twinr config."""

        primary_user_id = normalize_text(getattr(config, "portrait_match_primary_user_id", None)) or "main_user"
        temporal_window = _coerce_positive_float(
            getattr(config, "portrait_match_temporal_window_s", _DEFAULT_TEMPORAL_WINDOW_S),
            default=_DEFAULT_TEMPORAL_WINDOW_S,
        )
        portrait_max = _coerce_positive_int(
            getattr(config, "portrait_match_temporal_max_observations", _DEFAULT_MAX_OBSERVATIONS),
            default=_DEFAULT_MAX_OBSERVATIONS,
        )
        portrait_min = _coerce_positive_int(
            getattr(config, "portrait_match_temporal_min_observations", 2),
            default=2,
        )
        min_session_observations = max(_DEFAULT_MIN_SESSION_OBSERVATIONS, portrait_min)
        min_track_observations = max(_DEFAULT_MIN_TRACK_OBSERVATIONS, min_session_observations - 1)
        return cls(
            primary_user_id=primary_user_id,
            temporal_window_s=max(60.0, temporal_window),
            max_observations=max(_DEFAULT_MAX_OBSERVATIONS, portrait_max * 2),
            min_session_observations=min_session_observations,
            min_track_observations=min_track_observations,
            min_support_ratio=_DEFAULT_MIN_SUPPORT_RATIO,
        )


@dataclass(frozen=True, slots=True)
class MultimodalIdentityFusionSnapshot:
    """Describe one conservative temporal identity-fusion assessment."""

    observed_at: float | None = None
    state: str = "no_identity_signal"
    matches_main_user: bool = False
    matched_user_id: str | None = None
    matched_user_display_name: str | None = None
    policy_recommendation: str = "blocked"
    block_reason: str | None = None
    claim: RuntimeClaimMetadata = field(default_factory=_default_claim)
    temporal_state: str | None = None
    session_consistency_state: str | None = None
    session_observation_count: int | None = None
    session_support_ratio: float | None = None
    session_conflict_count: int | None = None
    track_consistency_state: str | None = None
    track_observation_count: int | None = None
    track_support_ratio: float | None = None
    track_anchor_zone: str | None = None
    presence_session_id: int | None = None
    voice_status: str | None = None
    voice_confidence: float | None = None
    voice_checked_age_s: float | None = None
    voice_matched_user_id: str | None = None
    voice_matched_user_display_name: str | None = None
    voice_match_source: str | None = None
    speaker_association_state: str | None = None
    speaker_association_confidence: float | None = None
    portrait_match_state: str | None = None
    portrait_match_confidence: float | None = None
    portrait_match_temporal_state: str | None = None
    portrait_match_fused_confidence: float | None = None
    portrait_match_observation_count: int | None = None

    def to_automation_facts(self) -> dict[str, object]:
        """Serialize the fusion snapshot into automation-friendly facts."""

        payload = {
            "observed_at": self.observed_at,
            "state": self.state,
            "matches_main_user": self.matches_main_user,
            "matched_user_id": self.matched_user_id,
            "matched_user_display_name": self.matched_user_display_name,
            "policy_recommendation": self.policy_recommendation,
            "block_reason": self.block_reason,
            "temporal_state": self.temporal_state,
            "session_consistency_state": self.session_consistency_state,
            "session_observation_count": self.session_observation_count,
            "session_support_ratio": self.session_support_ratio,
            "session_conflict_count": self.session_conflict_count,
            "track_consistency_state": self.track_consistency_state,
            "track_observation_count": self.track_observation_count,
            "track_support_ratio": self.track_support_ratio,
            "track_anchor_zone": self.track_anchor_zone,
            "presence_session_id": self.presence_session_id,
            "voice_status": self.voice_status,
            "voice_confidence": self.voice_confidence,
            "voice_checked_age_s": self.voice_checked_age_s,
            "voice_matched_user_id": self.voice_matched_user_id,
            "voice_matched_user_display_name": self.voice_matched_user_display_name,
            "voice_match_source": self.voice_match_source,
            "speaker_association_state": self.speaker_association_state,
            "speaker_association_confidence": self.speaker_association_confidence,
            "portrait_match_state": self.portrait_match_state,
            "portrait_match_confidence": self.portrait_match_confidence,
            "portrait_match_temporal_state": self.portrait_match_temporal_state,
            "portrait_match_fused_confidence": self.portrait_match_fused_confidence,
            "portrait_match_observation_count": self.portrait_match_observation_count,
        }
        payload.update(self.claim.to_payload())
        return payload

    def event_data(self) -> dict[str, object]:
        """Serialize the fusion snapshot into flat event fields."""

        return {
            "identity_fusion_state": self.state,
            "identity_fusion_matches_main_user": self.matches_main_user,
            "identity_fusion_confidence": self.claim.confidence,
            "identity_fusion_policy": self.policy_recommendation,
            "identity_fusion_temporal_state": self.temporal_state,
            "identity_fusion_matched_user_id": self.matched_user_id,
            "identity_fusion_voice_matched_user_id": self.voice_matched_user_id,
        }


@dataclass(frozen=True, slots=True)
class _IdentityFusionObservation:
    observed_at: float
    session_key: int | None
    candidate_user_id: str | None
    matches_main_user: bool
    conflict: bool
    visual_anchor_zone: str | None


class TemporalIdentityFusionTracker:
    """Keep bounded temporal identity evidence for the proactive runtime."""

    def __init__(self, *, config: MultimodalIdentityFusionConfig) -> None:
        self.config = config
        self._history: list[_IdentityFusionObservation] = []

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "TemporalIdentityFusionTracker":
        """Build one bounded tracker from global Twinr config."""

        return cls(config=MultimodalIdentityFusionConfig.from_config(config))

    def observe(
        self,
        *,
        observed_at: float | None,
        live_facts: Mapping[str, object] | None,
        voice_status: object | None,
        voice_confidence: object | None,
        voice_checked_at: object | None,
        voice_matched_user_id: object | None = None,
        voice_matched_user_display_name: object | None = None,
        voice_match_source: object | None = None,
        max_voice_age_s: int = _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S,
        presence_session_id: int | None = None,
        ambiguous_room_guard: AmbiguousRoomGuardSnapshot | None = None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None = None,
        portrait_match: PortraitMatchSnapshot | None = None,
        now_utc: datetime | None = None,
    ) -> MultimodalIdentityFusionSnapshot:
        """Return one conservative temporal identity-fusion snapshot."""

        guard = ambiguous_room_guard or derive_ambiguous_room_guard(
            observed_at=observed_at,
            live_facts=live_facts or {},
        )
        normalized_status = _normalize_voice_status(voice_status)
        normalized_voice_confidence = coerce_optional_ratio(voice_confidence)
        checked_age_s = _voice_checked_age_s(voice_checked_at, now_utc=now_utc)
        voice_fresh = (
            normalized_status is not None
            and checked_age_s is not None
            and checked_age_s <= max(1, int(max_voice_age_s))
        )
        voice_likely = normalized_status == "likely_user" and voice_fresh
        voice_other = normalized_status == "known_other_user" and voice_fresh
        voice_uncertain = normalized_status == "uncertain" and voice_fresh
        voice_uncertain_match = normalized_status == "uncertain_match" and voice_fresh
        voice_ambiguous = normalized_status == "ambiguous_match" and voice_fresh
        normalized_voice_user_id = normalize_text(voice_matched_user_id) or None
        normalized_voice_user_display_name = normalize_text(voice_matched_user_display_name) or None
        normalized_voice_match_source = normalize_text(voice_match_source) or None
        voice_candidate_user_id: str | None = None
        if voice_likely:
            voice_candidate_user_id = normalized_voice_user_id or self.config.primary_user_id
        elif voice_other or voice_uncertain_match:
            voice_candidate_user_id = normalized_voice_user_id
        elif voice_uncertain:
            voice_candidate_user_id = self.config.primary_user_id

        portrait_match_state = None if portrait_match is None else portrait_match.state
        portrait_match_confidence = None if portrait_match is None else portrait_match.claim.confidence
        portrait_match_temporal_state = None if portrait_match is None else portrait_match.temporal_state
        portrait_match_fused_confidence = None if portrait_match is None else portrait_match.fused_confidence
        portrait_match_observation_count = None if portrait_match is None else portrait_match.temporal_observation_count
        portrait_confidence = portrait_match_confidence
        if portrait_match_temporal_state == "stable_match" and portrait_match_fused_confidence is not None:
            portrait_confidence = portrait_match_fused_confidence

        speaker_association_state = None if speaker_association is None else speaker_association.state
        speaker_association_confidence = None if speaker_association is None else speaker_association.confidence
        current_zone = _camera_primary_person_zone(live_facts)

        if guard.guard_active:
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state="blocked_ambiguous_room",
                matches_main_user=False,
                matched_user_id=_matched_user_id(portrait_match),
                matched_user_display_name=_matched_user_display_name(portrait_match),
                policy_recommendation="blocked",
                block_reason=guard.reason,
                claim=RuntimeClaimMetadata(
                    confidence=guard.claim.confidence,
                    source="multimodal_identity_fusion_plus_ambiguous_room_guard",
                    requires_confirmation=True,
                ),
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        portrait_positive = portrait_match is not None and portrait_match.matches_reference_user
        portrait_other = portrait_match is not None and portrait_match.state == "known_other_user"
        portrait_uncertain_match = portrait_match is not None and portrait_match.state == "uncertain_match"
        portrait_conflict = portrait_match is not None and portrait_match.state in {
            "unknown_face",
            "known_other_user",
            "ambiguous_identity",
        }

        candidate_user_id: str | None = None
        matched_user_id = _matched_user_id(portrait_match)
        matched_display_name = _matched_user_display_name(portrait_match)
        matches_main_user = False
        conflict = False
        if portrait_other and matched_user_id:
            candidate_user_id = matched_user_id
            matched_display_name = matched_display_name or normalized_voice_user_display_name
            conflict = voice_candidate_user_id is not None and voice_candidate_user_id != matched_user_id
        elif portrait_positive:
            candidate_user_id = self.config.primary_user_id
            matched_display_name = matched_display_name or normalized_voice_user_display_name
            matches_main_user = voice_candidate_user_id == self.config.primary_user_id
            conflict = (voice_candidate_user_id is not None and voice_candidate_user_id != self.config.primary_user_id) or voice_ambiguous
        elif voice_candidate_user_id is not None:
            candidate_user_id = voice_candidate_user_id
            matched_display_name = normalized_voice_user_display_name or matched_display_name
            matches_main_user = voice_candidate_user_id == self.config.primary_user_id and voice_likely
            conflict = portrait_conflict or voice_ambiguous
        elif voice_ambiguous:
            conflict = True

        self._prune_history(observed_at)
        if candidate_user_id is not None or conflict:
            self._append_history(
                observed_at=observed_at,
                session_key=presence_session_id,
                candidate_user_id=candidate_user_id,
                matches_main_user=matches_main_user,
                conflict=conflict,
                visual_anchor_zone=current_zone,
            )

        session_observation_count, session_support_ratio, session_conflict_count, session_consistency_state = (
            self._session_consistency(
                session_key=presence_session_id,
                candidate_user_id=candidate_user_id,
            )
        )
        track_observation_count, track_support_ratio, track_consistency_state = self._track_consistency(
            session_key=presence_session_id,
            candidate_user_id=candidate_user_id,
            current_zone=current_zone,
            speaker_association=speaker_association,
        )
        temporal_state = _temporal_state(
            session_consistency_state=session_consistency_state,
            track_consistency_state=track_consistency_state,
        )

        if candidate_user_id is not None and candidate_user_id != self.config.primary_user_id and not conflict:
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state=(
                    "stable_other_enrolled_user"
                    if temporal_state == "stable_multimodal_match"
                    else "other_enrolled_user_candidate"
                ),
                matches_main_user=False,
                matched_user_id=candidate_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="blocked",
                block_reason="other_enrolled_user_detected",
                claim=RuntimeClaimMetadata(
                    confidence=_fusion_confidence(
                        normalized_voice_confidence,
                        portrait_confidence,
                        session_support_ratio,
                        track_support_ratio,
                        guard.claim.confidence,
                    ),
                    source=(
                        "household_voice_identity_plus_temporal_portrait_match_plus_track_history_plus_presence_session_memory"
                        if temporal_state == "stable_multimodal_match"
                        and normalized_voice_user_id == candidate_user_id
                        and portrait_other
                        else (
                            "temporal_portrait_match_plus_track_history_plus_presence_session_memory"
                            if temporal_state == "stable_multimodal_match" and portrait_other
                            else (
                                "household_voice_identity_plus_track_history_plus_presence_session_memory"
                                if normalized_voice_user_id == candidate_user_id
                                else "portrait_match_plus_clear_room_context"
                            )
                        )
                    ),
                    requires_confirmation=True,
                ),
                temporal_state=temporal_state,
                session_consistency_state=session_consistency_state,
                session_observation_count=session_observation_count,
                session_support_ratio=session_support_ratio,
                session_conflict_count=session_conflict_count,
                track_consistency_state=track_consistency_state,
                track_observation_count=track_observation_count,
                track_support_ratio=track_support_ratio,
                track_anchor_zone=current_zone,
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        if conflict:
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state="modality_conflict",
                matches_main_user=False,
                matched_user_id=candidate_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="confirm_first",
                block_reason="identity_modality_conflict",
                claim=RuntimeClaimMetadata(
                    confidence=_fusion_confidence(
                        normalized_voice_confidence,
                        portrait_confidence,
                        session_support_ratio,
                        guard.claim.confidence,
                    ),
                    source="voice_profile_plus_portrait_conflict_plus_presence_session_memory",
                    requires_confirmation=True,
                ),
                temporal_state=temporal_state,
                session_consistency_state=session_consistency_state,
                session_observation_count=session_observation_count,
                session_support_ratio=session_support_ratio,
                session_conflict_count=session_conflict_count,
                track_consistency_state=track_consistency_state,
                track_observation_count=track_observation_count,
                track_support_ratio=track_support_ratio,
                track_anchor_zone=current_zone,
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        if portrait_positive and voice_likely and temporal_state == "stable_multimodal_match":
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state="stable_main_user_multimodal",
                matches_main_user=True,
                matched_user_id=self.config.primary_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="calm_personalization_only",
                block_reason=None,
                claim=RuntimeClaimMetadata(
                    confidence=_fusion_confidence(
                        normalized_voice_confidence,
                        portrait_confidence,
                        session_support_ratio,
                        track_support_ratio,
                        guard.claim.confidence,
                    ),
                    source="voice_profile_plus_temporal_portrait_match_plus_track_history_plus_presence_session_memory",
                    requires_confirmation=True,
                ),
                temporal_state=temporal_state,
                session_consistency_state=session_consistency_state,
                session_observation_count=session_observation_count,
                session_support_ratio=session_support_ratio,
                session_conflict_count=session_conflict_count,
                track_consistency_state=track_consistency_state,
                track_observation_count=track_observation_count,
                track_support_ratio=track_support_ratio,
                track_anchor_zone=current_zone,
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        if portrait_positive and voice_likely:
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state="multimodal_candidate",
                matches_main_user=True,
                matched_user_id=self.config.primary_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="calm_personalization_only",
                block_reason=None,
                claim=RuntimeClaimMetadata(
                    confidence=_fusion_confidence(
                        normalized_voice_confidence,
                        portrait_confidence,
                        speaker_association_confidence,
                        guard.claim.confidence,
                    ),
                    source="voice_profile_plus_portrait_match_plus_clear_room_context",
                    requires_confirmation=True,
                ),
                temporal_state=temporal_state,
                session_consistency_state=session_consistency_state,
                session_observation_count=session_observation_count,
                session_support_ratio=session_support_ratio,
                session_conflict_count=session_conflict_count,
                track_consistency_state=track_consistency_state,
                track_observation_count=track_observation_count,
                track_support_ratio=track_support_ratio,
                track_anchor_zone=current_zone,
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        if portrait_positive or portrait_uncertain_match:
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state="portrait_session_candidate",
                matches_main_user=False,
                matched_user_id=self.config.primary_user_id if portrait_positive else None,
                matched_user_display_name=matched_display_name,
                policy_recommendation="confirm_first",
                block_reason="voice_signal_unavailable" if not voice_fresh else "voice_signal_uncertain",
                claim=RuntimeClaimMetadata(
                    confidence=_fusion_confidence(
                        portrait_confidence,
                        session_support_ratio,
                        track_support_ratio,
                        guard.claim.confidence,
                    ),
                    source="temporal_portrait_match_plus_track_history_plus_presence_session_memory",
                    requires_confirmation=True,
                ),
                temporal_state=temporal_state,
                session_consistency_state=session_consistency_state,
                session_observation_count=session_observation_count,
                session_support_ratio=session_support_ratio,
                session_conflict_count=session_conflict_count,
                track_consistency_state=track_consistency_state,
                track_observation_count=track_observation_count,
                track_support_ratio=track_support_ratio,
                track_anchor_zone=current_zone,
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        if voice_likely or voice_uncertain:
            return MultimodalIdentityFusionSnapshot(
                observed_at=observed_at,
                state="voice_session_candidate",
                matches_main_user=voice_likely,
                matched_user_id=self.config.primary_user_id,
                policy_recommendation="calm_personalization_only" if voice_likely else "confirm_first",
                block_reason=None if voice_likely else "voice_signal_uncertain",
                claim=RuntimeClaimMetadata(
                    confidence=_fusion_confidence(
                        normalized_voice_confidence,
                        session_support_ratio,
                        track_support_ratio,
                        guard.claim.confidence,
                    ),
                    source="voice_profile_plus_track_history_plus_presence_session_memory",
                    requires_confirmation=True,
                ),
                temporal_state=temporal_state,
                session_consistency_state=session_consistency_state,
                session_observation_count=session_observation_count,
                session_support_ratio=session_support_ratio,
                session_conflict_count=session_conflict_count,
                track_consistency_state=track_consistency_state,
                track_observation_count=track_observation_count,
                track_support_ratio=track_support_ratio,
                track_anchor_zone=current_zone,
                presence_session_id=presence_session_id,
                voice_status=normalized_status,
                voice_confidence=normalized_voice_confidence,
                voice_checked_age_s=checked_age_s,
                voice_matched_user_id=normalized_voice_user_id,
                voice_matched_user_display_name=normalized_voice_user_display_name,
                voice_match_source=normalized_voice_match_source,
                speaker_association_state=speaker_association_state,
                speaker_association_confidence=speaker_association_confidence,
                portrait_match_state=portrait_match_state,
                portrait_match_confidence=portrait_match_confidence,
                portrait_match_temporal_state=portrait_match_temporal_state,
                portrait_match_fused_confidence=portrait_match_fused_confidence,
                portrait_match_observation_count=portrait_match_observation_count,
            )

        return MultimodalIdentityFusionSnapshot(
            observed_at=observed_at,
            state="no_identity_signal",
            matches_main_user=False,
            matched_user_id=matched_user_id,
            matched_user_display_name=matched_display_name,
            policy_recommendation="blocked",
            block_reason="identity_signal_unavailable",
            claim=RuntimeClaimMetadata(
                confidence=0.0,
                source="multimodal_identity_fusion",
                requires_confirmation=True,
            ),
            presence_session_id=presence_session_id,
            voice_status=normalized_status,
            voice_confidence=normalized_voice_confidence,
            voice_checked_age_s=checked_age_s,
            voice_matched_user_id=normalized_voice_user_id,
            voice_matched_user_display_name=normalized_voice_user_display_name,
            voice_match_source=normalized_voice_match_source,
            speaker_association_state=speaker_association_state,
            speaker_association_confidence=speaker_association_confidence,
            portrait_match_state=portrait_match_state,
            portrait_match_confidence=portrait_match_confidence,
            portrait_match_temporal_state=portrait_match_temporal_state,
            portrait_match_fused_confidence=portrait_match_fused_confidence,
            portrait_match_observation_count=portrait_match_observation_count,
        )

    def _append_history(
        self,
        *,
        observed_at: float | None,
        session_key: int | None,
        candidate_user_id: str | None,
        matches_main_user: bool,
        conflict: bool,
        visual_anchor_zone: str | None,
    ) -> None:
        checked_at = 0.0 if observed_at is None else float(observed_at)
        entry = _IdentityFusionObservation(
            observed_at=checked_at,
            session_key=session_key,
            candidate_user_id=candidate_user_id,
            matches_main_user=matches_main_user,
            conflict=conflict,
            visual_anchor_zone=visual_anchor_zone,
        )
        if self._history and self._history[-1] == entry:
            return
        self._history.append(entry)
        if len(self._history) > self.config.max_observations:
            self._history = self._history[-self.config.max_observations :]

    def _prune_history(self, observed_at: float | None) -> None:
        checked_at = 0.0 if observed_at is None else float(observed_at)
        minimum_time = checked_at - self.config.temporal_window_s
        self._history = [item for item in self._history if item.observed_at >= minimum_time]
        if len(self._history) > self.config.max_observations:
            self._history = self._history[-self.config.max_observations :]

    def _session_consistency(
        self,
        *,
        session_key: int | None,
        candidate_user_id: str | None,
    ) -> tuple[int | None, float | None, int | None, str | None]:
        if candidate_user_id is None:
            return None, None, None, None
        same_session = [item for item in self._history if item.session_key == session_key and item.candidate_user_id]
        if not same_session:
            return None, None, None, None
        support = [item for item in same_session if item.candidate_user_id == candidate_user_id]
        conflict_count = sum(
            1
            for item in same_session
            if item.conflict or (item.candidate_user_id is not None and item.candidate_user_id != candidate_user_id)
        )
        support_ratio = round(len(support) / max(1, len(same_session)), 4)
        if conflict_count > 0:
            state = "recent_conflict"
        elif (
            len(support) >= self.config.min_session_observations
            and support_ratio >= self.config.min_support_ratio
        ):
            state = "stable_session"
        else:
            state = "insufficient_history"
        return len(support), support_ratio, conflict_count, state

    def _track_consistency(
        self,
        *,
        session_key: int | None,
        candidate_user_id: str | None,
        current_zone: str | None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
    ) -> tuple[int | None, float | None, str | None]:
        if speaker_association is not None and speaker_association.associated and speaker_association.confidence is not None:
            return 1, speaker_association.confidence, "speaker_locked"
        if candidate_user_id is None or not current_zone:
            return None, None, None
        same_session = [
            item
            for item in self._history
            if item.session_key == session_key
            and item.candidate_user_id == candidate_user_id
            and item.visual_anchor_zone
        ]
        if not same_session:
            return None, None, None
        support = [item for item in same_session if item.visual_anchor_zone == current_zone]
        support_ratio = round(len(support) / max(1, len(same_session)), 4)
        if (
            len(same_session) >= self.config.min_track_observations
            and support_ratio >= self.config.min_support_ratio
        ):
            return len(same_session), support_ratio, "stable_anchor"
        if len(same_session) > 1:
            return len(same_session), support_ratio, "recent_anchor_shift"
        return len(same_session), support_ratio, "insufficient_history"


def _matched_user_id(portrait_match: PortraitMatchSnapshot | None) -> str | None:
    return None if portrait_match is None else normalize_text(portrait_match.matched_user_id) or None


def _matched_user_display_name(portrait_match: PortraitMatchSnapshot | None) -> str | None:
    return None if portrait_match is None else normalize_text(portrait_match.matched_user_display_name) or None


def _camera_primary_person_zone(live_facts: Mapping[str, object] | None) -> str | None:
    if not isinstance(live_facts, Mapping):
        return None
    camera = live_facts.get("camera")
    if not isinstance(camera, Mapping):
        return None
    value = normalize_text(camera.get("primary_person_zone")).lower()
    if value in {"left", "center", "right"}:
        return value
    return None


def _normalize_voice_status(value: object | None) -> str | None:
    normalized = normalize_text(value).lower()
    if not normalized:
        return None
    if normalized in _ALLOWED_VOICE_STATUSES:
        return normalized
    return "unknown_voice"


def _voice_checked_age_s(value: object | None, *, now_utc: datetime | None) -> float | None:
    parsed = _parse_checked_at(value)
    if parsed is None:
        return None
    effective_now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    if parsed > effective_now + timedelta(seconds=_MAX_FUTURE_SKEW_S):
        return None
    return round(max(0.0, (effective_now - parsed).total_seconds()), 3)


def _parse_checked_at(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = normalize_text(value)
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = f"{raw[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _fusion_confidence(*values: float | None) -> float:
    mean = mean_confidence(tuple(values))
    if mean is None:
        return 0.0
    return round(max(0.0, min(1.0, mean)), 4)


def _temporal_state(*, session_consistency_state: str | None, track_consistency_state: str | None) -> str | None:
    if session_consistency_state == "stable_session" and track_consistency_state in {"stable_anchor", "speaker_locked"}:
        return "stable_multimodal_match"
    if session_consistency_state == "recent_conflict" or track_consistency_state == "recent_anchor_shift":
        return "recent_conflict"
    if session_consistency_state is None and track_consistency_state is None:
        return None
    return "insufficient_history"


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0.0:
        return default
    return parsed


def _coerce_positive_int(value: object, *, default: int) -> int:
    parsed = coerce_optional_int(value)
    if parsed is None or parsed < 1:
        return default
    return parsed


__all__ = [
    "MultimodalIdentityFusionConfig",
    "MultimodalIdentityFusionSnapshot",
    "TemporalIdentityFusionTracker",
]
