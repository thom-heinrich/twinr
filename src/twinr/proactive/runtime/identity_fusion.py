# CHANGELOG: 2026-03-29
# BUG-1: Fixed stale-history bleed when observed_at/checked_at was missing by using current event time
#        as a bounded fallback instead of 0.0, and by pruning against real time.
# BUG-2: Fixed frame-rate bias where near-identical consecutive observations could reach
#        "stable_*" states after a few adjacent frames; repeated observations are now coalesced
#        inside a short dedup window.
# BUG-3: Fixed a real race condition around shared temporal memory by protecting all compound
#        reads/writes with an internal lock.
# SEC-1: Added optional spoof/liveness gating for audio and visual identity hints so replayed,
#        deepfake, or suspicious media can block or downgrade identity decisions.
# SEC-2: Added derived fallback presence-session segmentation when upstream does not provide a
#        session id, preventing cross-occupant evidence carry-over on long-running Pi deployments.
# IMP-1: Replaced naive mean score fusion with quality-aware, uncertainty-aware weighted fusion
#        inspired by 2025/2026 multimodal biometric fusion practice.
# IMP-2: Upgraded temporal fusion with recency decay, optional track-id anchoring, and explicit
#        abstain/confirm-first behavior for open-set or security-uncertain cases.
# BREAKING: observe() can now return state="blocked_spoof_risk" when anti-spoof or liveness
#           signals indicate replay/deepfake/attack risk.
# BREAKING: voice-only positive evidence no longer escalates to calm_personalization_only;
#           it stays confirm_first unless a corroborating visual match is present.
# BREAKING: presence_session_id now exposes the effective fusion session key; when the upstream
#           pipeline omits a session id, a derived negative key is emitted and the original value
#           is preserved in input_presence_session_id.

"""Fuse live identity hints with bounded temporal and session memory.

This module keeps the stateful, short-lived identity evidence tracker out of
``service.py`` and ``known_user_hint.py``. It combines conservative voice
verification, local portrait matching, speaker association, visual-anchor
history, optional anti-spoof/liveness signals, and presence-session scoped
memory into one confirm-first runtime surface.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import isfinite
from threading import RLock

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.ambiguous_room_guard import (
    AmbiguousRoomGuardSnapshot,
    derive_ambiguous_room_guard,
)
from twinr.proactive.runtime.claim_metadata import (
    RuntimeClaimMetadata,
    coerce_optional_int,
    coerce_optional_ratio,
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
_DEFAULT_GUARD_ASSESSMENT_MAX_AGE_S = 120
_DEFAULT_SESSION_RESET_GAP_S = 45.0
_DEFAULT_OBSERVATION_DEDUP_WINDOW_S = 0.75
_DEFAULT_EVIDENCE_HALF_LIFE_S = 25.0
_MAX_FUTURE_SKEW_S = 5.0
_MAX_SAFE_TEXT_LEN = 96
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
_ALLOWED_SECURITY_STATUSES = frozenset(
    {
        "live",
        "likely_live",
        "bona_fide",
        "passed",
        "unknown",
        "uncertain",
        "suspected_spoof",
        "spoof",
        "replay",
        "deepfake",
        "attack",
    }
)
_BLOCKING_SECURITY_STATUSES = frozenset({"spoof", "replay", "deepfake", "attack"})
_SOFT_SECURITY_STATUSES = frozenset({"unknown", "uncertain", "suspected_spoof"})
_POSITIVE_SECURITY_STATUSES = frozenset({"live", "likely_live", "bona_fide", "passed"})
_STABLE_TRACK_STATES = frozenset({"stable_anchor", "stable_track", "speaker_locked"})
_CONFLICT_TRACK_STATES = frozenset({"recent_anchor_shift", "recent_track_shift"})


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
    session_reset_gap_s: float = _DEFAULT_SESSION_RESET_GAP_S
    observation_dedup_window_s: float = _DEFAULT_OBSERVATION_DEDUP_WINDOW_S
    evidence_half_life_s: float = _DEFAULT_EVIDENCE_HALF_LIFE_S
    max_voice_age_s: int = _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S
    max_guard_age_s: int = _DEFAULT_GUARD_ASSESSMENT_MAX_AGE_S

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "MultimodalIdentityFusionConfig":
        """Derive one bounded fusion config from the global Twinr config."""

        primary_user_id = _bounded_text(getattr(config, "portrait_match_primary_user_id", None)) or "main_user"
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
        session_reset_gap_s = _coerce_positive_float(
            getattr(config, "multimodal_identity_session_reset_gap_s", _DEFAULT_SESSION_RESET_GAP_S),
            default=_DEFAULT_SESSION_RESET_GAP_S,
        )
        dedup_window_s = _coerce_positive_float(
            getattr(config, "multimodal_identity_observation_dedup_window_s", _DEFAULT_OBSERVATION_DEDUP_WINDOW_S),
            default=_DEFAULT_OBSERVATION_DEDUP_WINDOW_S,
        )
        evidence_half_life_s = _coerce_positive_float(
            getattr(config, "multimodal_identity_evidence_half_life_s", _DEFAULT_EVIDENCE_HALF_LIFE_S),
            default=_DEFAULT_EVIDENCE_HALF_LIFE_S,
        )
        max_voice_age_s = _coerce_positive_int(
            getattr(config, "multimodal_identity_voice_max_age_s", _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S),
            default=_DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S,
        )
        max_guard_age_s = _coerce_positive_int(
            getattr(config, "multimodal_identity_guard_max_age_s", _DEFAULT_GUARD_ASSESSMENT_MAX_AGE_S),
            default=_DEFAULT_GUARD_ASSESSMENT_MAX_AGE_S,
        )
        return cls(
            primary_user_id=primary_user_id,
            temporal_window_s=max(60.0, temporal_window),
            max_observations=max(_DEFAULT_MAX_OBSERVATIONS, portrait_max * 2),
            min_session_observations=min_session_observations,
            min_track_observations=min_track_observations,
            min_support_ratio=_DEFAULT_MIN_SUPPORT_RATIO,
            session_reset_gap_s=max(10.0, min(max(temporal_window, 10.0), session_reset_gap_s)),
            observation_dedup_window_s=min(max(0.1, dedup_window_s), 5.0),
            evidence_half_life_s=max(5.0, min(temporal_window, evidence_half_life_s)),
            max_voice_age_s=max(1, max_voice_age_s),
            max_guard_age_s=max(1, max_guard_age_s),
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
    track_anchor_key: str | None = None
    presence_session_id: int | None = None
    input_presence_session_id: int | None = None
    voice_status: str | None = None
    voice_confidence: float | None = None
    voice_checked_age_s: float | None = None
    voice_matched_user_id: str | None = None
    voice_matched_user_display_name: str | None = None
    voice_match_source: str | None = None
    voice_spoof_state: str | None = None
    voice_spoof_confidence: float | None = None
    voice_spoof_risk: float | None = None
    voice_spoof_checked_age_s: float | None = None
    face_liveness_state: str | None = None
    face_liveness_confidence: float | None = None
    face_spoof_risk: float | None = None
    face_liveness_checked_age_s: float | None = None
    voice_reliability: float | None = None
    portrait_reliability: float | None = None
    temporal_reliability: float | None = None
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
            "track_anchor_key": self.track_anchor_key,
            "presence_session_id": self.presence_session_id,
            "input_presence_session_id": self.input_presence_session_id,
            "voice_status": self.voice_status,
            "voice_confidence": self.voice_confidence,
            "voice_checked_age_s": self.voice_checked_age_s,
            "voice_matched_user_id": self.voice_matched_user_id,
            "voice_matched_user_display_name": self.voice_matched_user_display_name,
            "voice_match_source": self.voice_match_source,
            "voice_spoof_state": self.voice_spoof_state,
            "voice_spoof_confidence": self.voice_spoof_confidence,
            "voice_spoof_risk": self.voice_spoof_risk,
            "voice_spoof_checked_age_s": self.voice_spoof_checked_age_s,
            "face_liveness_state": self.face_liveness_state,
            "face_liveness_confidence": self.face_liveness_confidence,
            "face_spoof_risk": self.face_spoof_risk,
            "face_liveness_checked_age_s": self.face_liveness_checked_age_s,
            "voice_reliability": self.voice_reliability,
            "portrait_reliability": self.portrait_reliability,
            "temporal_reliability": self.temporal_reliability,
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
            "identity_fusion_presence_session_id": self.presence_session_id,
            "identity_fusion_voice_spoof_state": self.voice_spoof_state,
            "identity_fusion_face_liveness_state": self.face_liveness_state,
        }


@dataclass(frozen=True, slots=True)
class _IdentityFusionObservation:
    observed_at: float
    session_key: int | None
    candidate_user_id: str | None
    matches_main_user: bool
    conflict: bool
    visual_anchor_zone: str | None
    visual_track_key: str | None
    evidence_weight: float


class TemporalIdentityFusionTracker:
    """Keep bounded temporal identity evidence for the proactive runtime."""

    def __init__(self, *, config: MultimodalIdentityFusionConfig) -> None:
        self.config = config
        self._history: deque[_IdentityFusionObservation] = deque(maxlen=config.max_observations)
        self._lock = RLock()
        self._fallback_session_counter = 0
        self._active_fallback_session_key: int | None = None
        self._active_fallback_session_seen_at: float | None = None

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
        voice_spoof_status: object | None = None,
        voice_spoof_confidence: object | None = None,
        voice_spoof_checked_at: object | None = None,
        voice_spoof_risk: object | None = None,
        face_liveness_status: object | None = None,
        face_liveness_confidence: object | None = None,
        face_liveness_checked_at: object | None = None,
        face_spoof_risk: object | None = None,
    ) -> MultimodalIdentityFusionSnapshot:
        """Return one conservative temporal identity-fusion snapshot."""

        live_facts_map: Mapping[str, object] = live_facts or {}
        effective_observed_at = _effective_event_time(observed_at, now_utc=now_utc)
        audio_facts = _nested_mapping(live_facts_map, "audio")
        camera_facts = _nested_mapping(live_facts_map, "camera")

        normalized_status = _normalize_voice_status(voice_status)
        normalized_voice_confidence = coerce_optional_ratio(voice_confidence)
        checked_age_s = _checked_age_s(
            voice_checked_at,
            now_utc=now_utc,
            fallback_observed_at=effective_observed_at,
        )
        requested_max_voice_age_s = coerce_optional_int(max_voice_age_s)
        if (
            requested_max_voice_age_s is None
            or (
                requested_max_voice_age_s == _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S
                and self.config.max_voice_age_s != _DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S
            )
        ):
            effective_max_voice_age_s = self.config.max_voice_age_s
        else:
            effective_max_voice_age_s = max(1, requested_max_voice_age_s)
        voice_fresh = (
            normalized_status is not None
            and checked_age_s is not None
            and checked_age_s <= effective_max_voice_age_s
        )
        voice_likely = normalized_status == "likely_user" and voice_fresh
        voice_other = normalized_status == "known_other_user" and voice_fresh
        voice_uncertain = normalized_status == "uncertain" and voice_fresh
        voice_uncertain_match = normalized_status == "uncertain_match" and voice_fresh
        voice_ambiguous = normalized_status == "ambiguous_match" and voice_fresh
        normalized_voice_user_id = _bounded_text(voice_matched_user_id)
        normalized_voice_user_display_name = _bounded_text(voice_matched_user_display_name)
        normalized_voice_match_source = _bounded_text(voice_match_source)

        voice_spoof_state = _normalize_security_status(
            voice_spoof_status
            if voice_spoof_status is not None
            else _coalesce_value(
                audio_facts,
                "voice_spoof_status",
                "speaker_spoof_status",
                "asv_spoof_status",
                "voice_liveness_status",
            )
        )
        voice_spoof_confidence_value = coerce_optional_ratio(
            voice_spoof_confidence
            if voice_spoof_confidence is not None
            else _coalesce_value(
                audio_facts,
                "voice_spoof_confidence",
                "speaker_spoof_confidence",
                "voice_liveness_confidence",
            )
        )
        voice_spoof_risk_value = coerce_optional_ratio(
            voice_spoof_risk
            if voice_spoof_risk is not None
            else _coalesce_value(
                audio_facts,
                "voice_spoof_risk",
                "speaker_spoof_risk",
                "voice_attack_risk",
            )
        )
        voice_spoof_checked_age_s = _checked_age_s(
            voice_spoof_checked_at
            if voice_spoof_checked_at is not None
            else _coalesce_value(
                audio_facts,
                "voice_spoof_checked_at",
                "speaker_spoof_checked_at",
                "voice_liveness_checked_at",
            ),
            now_utc=now_utc,
            fallback_observed_at=effective_observed_at
            if voice_spoof_state is not None or voice_spoof_risk_value is not None or voice_spoof_confidence_value is not None
            else None,
        )
        voice_spoof_fresh = (
            (voice_spoof_state is not None or voice_spoof_risk_value is not None or voice_spoof_confidence_value is not None)
            and voice_spoof_checked_age_s is not None
            and voice_spoof_checked_age_s <= self.config.max_guard_age_s
        )
        voice_spoof_blocked, voice_spoof_uncertain = _security_posture(
            status=voice_spoof_state,
            risk=voice_spoof_risk_value,
            fresh=voice_spoof_fresh,
        )

        face_liveness_state_value = _normalize_security_status(
            face_liveness_status
            if face_liveness_status is not None
            else _coalesce_value(
                camera_facts,
                "face_liveness_status",
                "visual_liveness_status",
                "face_spoof_status",
            )
        )
        face_liveness_confidence_value = coerce_optional_ratio(
            face_liveness_confidence
            if face_liveness_confidence is not None
            else _coalesce_value(
                camera_facts,
                "face_liveness_confidence",
                "visual_liveness_confidence",
                "face_spoof_confidence",
            )
        )
        face_spoof_risk_value = coerce_optional_ratio(
            face_spoof_risk
            if face_spoof_risk is not None
            else _coalesce_value(
                camera_facts,
                "face_spoof_risk",
                "visual_spoof_risk",
            )
        )
        face_liveness_checked_age_s = _checked_age_s(
            face_liveness_checked_at
            if face_liveness_checked_at is not None
            else _coalesce_value(
                camera_facts,
                "face_liveness_checked_at",
                "visual_liveness_checked_at",
                "face_spoof_checked_at",
            ),
            now_utc=now_utc,
            fallback_observed_at=effective_observed_at
            if face_liveness_state_value is not None
            or face_liveness_confidence_value is not None
            or face_spoof_risk_value is not None
            else None,
        )
        face_liveness_fresh = (
            (face_liveness_state_value is not None or face_liveness_confidence_value is not None or face_spoof_risk_value is not None)
            and face_liveness_checked_age_s is not None
            and face_liveness_checked_age_s <= self.config.max_guard_age_s
        )
        face_spoof_blocked, face_spoof_uncertain = _security_posture(
            status=face_liveness_state_value,
            risk=face_spoof_risk_value,
            fresh=face_liveness_fresh,
        )

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
        current_zone = _camera_primary_person_zone(live_facts_map)
        current_track_key = _camera_primary_person_track_key(live_facts_map, speaker_association=speaker_association)

        with self._lock:
            resolved_session_key = self._resolve_session_key(
                provided_session_key=presence_session_id,
                observed_at=effective_observed_at,
            )

        guard = ambiguous_room_guard or derive_ambiguous_room_guard(
            observed_at=effective_observed_at,
            live_facts=live_facts_map,
        )

        matched_user_id = _matched_user_id(portrait_match)
        matched_display_name = _matched_user_display_name(portrait_match)

        voice_reliability = _voice_reliability(
            status=normalized_status,
            confidence=normalized_voice_confidence,
            fresh=voice_fresh,
            spoof_blocked=voice_spoof_blocked,
            spoof_uncertain=voice_spoof_uncertain,
            spoof_status=voice_spoof_state,
        )
        portrait_reliability = _portrait_reliability(
            portrait_state=portrait_match_state,
            portrait_confidence=portrait_confidence,
            face_spoof_blocked=face_spoof_blocked,
            face_spoof_uncertain=face_spoof_uncertain,
            face_liveness_state=face_liveness_state_value,
        )

        common_fields: dict[str, object] = {
            "observed_at": effective_observed_at,
            "presence_session_id": resolved_session_key,
            "input_presence_session_id": presence_session_id,
            "voice_status": normalized_status,
            "voice_confidence": normalized_voice_confidence,
            "voice_checked_age_s": checked_age_s,
            "voice_matched_user_id": normalized_voice_user_id,
            "voice_matched_user_display_name": normalized_voice_user_display_name,
            "voice_match_source": normalized_voice_match_source,
            "voice_spoof_state": voice_spoof_state,
            "voice_spoof_confidence": voice_spoof_confidence_value,
            "voice_spoof_risk": voice_spoof_risk_value,
            "voice_spoof_checked_age_s": voice_spoof_checked_age_s,
            "face_liveness_state": face_liveness_state_value,
            "face_liveness_confidence": face_liveness_confidence_value,
            "face_spoof_risk": face_spoof_risk_value,
            "face_liveness_checked_age_s": face_liveness_checked_age_s,
            "voice_reliability": voice_reliability,
            "portrait_reliability": portrait_reliability,
            "speaker_association_state": speaker_association_state,
            "speaker_association_confidence": speaker_association_confidence,
            "portrait_match_state": portrait_match_state,
            "portrait_match_confidence": portrait_match_confidence,
            "portrait_match_temporal_state": portrait_match_temporal_state,
            "portrait_match_fused_confidence": portrait_match_fused_confidence,
            "portrait_match_observation_count": portrait_match_observation_count,
            "track_anchor_zone": current_zone,
            "track_anchor_key": current_track_key,
        }

        def build_snapshot(**overrides: object) -> MultimodalIdentityFusionSnapshot:
            payload = dict(common_fields)
            payload.update(overrides)
            return MultimodalIdentityFusionSnapshot(**payload)

        if voice_spoof_blocked or face_spoof_blocked:
            security_block_reason = "multimodal_spoof_risk"
            if voice_spoof_blocked and not face_spoof_blocked:
                security_block_reason = "voice_spoof_detected"
            elif face_spoof_blocked and not voice_spoof_blocked:
                security_block_reason = "face_spoof_detected"
            return build_snapshot(
                state="blocked_spoof_risk",
                matches_main_user=False,
                matched_user_id=matched_user_id or normalized_voice_user_id,
                matched_user_display_name=matched_display_name or normalized_voice_user_display_name,
                policy_recommendation="blocked",
                block_reason=security_block_reason,
                claim=RuntimeClaimMetadata(
                    confidence=_weighted_confidence(
                        (normalized_voice_confidence, voice_reliability),
                        (portrait_confidence, portrait_reliability),
                        (voice_spoof_risk_value, 1.0),
                        (face_spoof_risk_value, 1.0),
                    ),
                    source="multimodal_identity_fusion_plus_spoof_guard",
                    requires_confirmation=True,
                ),
                temporal_reliability=0.0,
            )

        if guard.guard_active:
            return build_snapshot(
                state="blocked_ambiguous_room",
                matches_main_user=False,
                matched_user_id=matched_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="blocked",
                block_reason=guard.reason,
                claim=RuntimeClaimMetadata(
                    confidence=guard.claim.confidence,
                    source="multimodal_identity_fusion_plus_ambiguous_room_guard",
                    requires_confirmation=True,
                ),
                temporal_reliability=0.0,
            )

        voice_candidate_user_id: str | None = None
        if voice_likely:
            voice_candidate_user_id = normalized_voice_user_id or self.config.primary_user_id
        elif voice_other or voice_uncertain_match:
            voice_candidate_user_id = normalized_voice_user_id
        elif voice_uncertain:
            voice_candidate_user_id = self.config.primary_user_id

        portrait_positive = portrait_match is not None and portrait_match.matches_reference_user and not face_spoof_blocked
        portrait_other = portrait_match is not None and portrait_match.state == "known_other_user" and not face_spoof_blocked
        portrait_uncertain_match = portrait_match is not None and portrait_match.state == "uncertain_match" and not face_spoof_blocked
        portrait_conflict = portrait_match is not None and portrait_match.state in {
            "unknown_face",
            "known_other_user",
            "ambiguous_identity",
        }
        if face_spoof_uncertain:
            portrait_conflict = portrait_conflict or portrait_positive or portrait_other

        candidate_user_id: str | None = None
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

        observation_weight = _weighted_confidence(
            (normalized_voice_confidence, voice_reliability),
            (portrait_confidence, portrait_reliability),
            (speaker_association_confidence, 0.25),
        )

        with self._lock:
            self._prune_history(observed_at=effective_observed_at)
            if candidate_user_id is not None or conflict:
                self._append_history(
                    observed_at=effective_observed_at,
                    session_key=resolved_session_key,
                    candidate_user_id=candidate_user_id,
                    matches_main_user=matches_main_user,
                    conflict=conflict,
                    visual_anchor_zone=current_zone,
                    visual_track_key=current_track_key,
                    evidence_weight=max(0.15, observation_weight) if (candidate_user_id is not None or conflict) else 0.0,
                )
            (
                session_observation_count,
                session_support_ratio,
                session_conflict_count,
                session_consistency_state,
            ) = self._session_consistency(
                session_key=resolved_session_key,
                candidate_user_id=candidate_user_id,
                observed_at=effective_observed_at,
            )
            (
                track_observation_count,
                track_support_ratio,
                track_consistency_state,
            ) = self._track_consistency(
                session_key=resolved_session_key,
                candidate_user_id=candidate_user_id,
                current_zone=current_zone,
                current_track_key=current_track_key,
                speaker_association=speaker_association,
                observed_at=effective_observed_at,
            )

        temporal_state = _temporal_state(
            session_consistency_state=session_consistency_state,
            track_consistency_state=track_consistency_state,
        )
        temporal_reliability = _temporal_reliability(
            session_support_ratio=session_support_ratio,
            track_support_ratio=track_support_ratio,
            temporal_state=temporal_state,
            speaker_association_confidence=speaker_association_confidence,
        )

        security_confirmation_required = (
            voice_spoof_uncertain
            or face_spoof_uncertain
            or (voice_likely and not portrait_positive and not portrait_other)
        )
        voice_signal_reason = (
            "voice_signal_unavailable"
            if not voice_fresh
            else "voice_signal_uncertain"
            if normalized_status in {"uncertain", "uncertain_match", "ambiguous_match"}
            else None
        )

        common_decision_fields: dict[str, object] = {
            "temporal_state": temporal_state,
            "session_consistency_state": session_consistency_state,
            "session_observation_count": session_observation_count,
            "session_support_ratio": session_support_ratio,
            "session_conflict_count": session_conflict_count,
            "track_consistency_state": track_consistency_state,
            "track_observation_count": track_observation_count,
            "track_support_ratio": track_support_ratio,
            "temporal_reliability": temporal_reliability,
        }

        if candidate_user_id is not None and candidate_user_id != self.config.primary_user_id and not conflict:
            return build_snapshot(
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
                    confidence=_weighted_confidence(
                        (normalized_voice_confidence, voice_reliability),
                        (portrait_confidence, portrait_reliability),
                        (session_support_ratio, 0.45),
                        (track_support_ratio, 0.35),
                    ),
                    source="quality_aware_temporal_identity_fusion",
                    requires_confirmation=True,
                ),
                **common_decision_fields,
            )

        if conflict:
            return build_snapshot(
                state="modality_conflict",
                matches_main_user=False,
                matched_user_id=candidate_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="confirm_first",
                block_reason="identity_modality_conflict",
                claim=RuntimeClaimMetadata(
                    confidence=_weighted_confidence(
                        (normalized_voice_confidence, voice_reliability),
                        (portrait_confidence, portrait_reliability),
                        (session_support_ratio, 0.35),
                    ),
                    source="quality_aware_conflict_sensitive_identity_fusion",
                    requires_confirmation=True,
                ),
                **common_decision_fields,
            )

        if (
            portrait_positive
            and voice_likely
            and temporal_state == "stable_multimodal_match"
            and not security_confirmation_required
        ):
            return build_snapshot(
                state="stable_main_user_multimodal",
                matches_main_user=True,
                matched_user_id=self.config.primary_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="calm_personalization_only",
                block_reason=None,
                claim=RuntimeClaimMetadata(
                    confidence=_weighted_confidence(
                        (normalized_voice_confidence, voice_reliability),
                        (portrait_confidence, portrait_reliability),
                        (session_support_ratio, 0.45),
                        (track_support_ratio, 0.4),
                    ),
                    source="quality_aware_temporal_multimodal_identity_fusion",
                    requires_confirmation=True,
                ),
                **common_decision_fields,
            )

        if portrait_positive and voice_likely:
            return build_snapshot(
                state="multimodal_candidate",
                matches_main_user=True,
                matched_user_id=self.config.primary_user_id,
                matched_user_display_name=matched_display_name,
                policy_recommendation="confirm_first" if security_confirmation_required else "calm_personalization_only",
                block_reason=(
                    "spoof_guard_uncertain"
                    if voice_spoof_uncertain or face_spoof_uncertain
                    else None
                ),
                claim=RuntimeClaimMetadata(
                    confidence=_weighted_confidence(
                        (normalized_voice_confidence, voice_reliability),
                        (portrait_confidence, portrait_reliability),
                        (speaker_association_confidence, 0.25),
                        (session_support_ratio, 0.2),
                    ),
                    source="quality_aware_multimodal_identity_fusion",
                    requires_confirmation=True,
                ),
                **common_decision_fields,
            )

        if portrait_positive or portrait_uncertain_match:
            return build_snapshot(
                state="portrait_session_candidate",
                matches_main_user=False,
                matched_user_id=self.config.primary_user_id if portrait_positive else None,
                matched_user_display_name=matched_display_name,
                policy_recommendation="confirm_first",
                block_reason=voice_signal_reason or ("spoof_guard_uncertain" if face_spoof_uncertain else None),
                claim=RuntimeClaimMetadata(
                    confidence=_weighted_confidence(
                        (portrait_confidence, portrait_reliability),
                        (session_support_ratio, 0.45),
                        (track_support_ratio, 0.35),
                    ),
                    source="quality_aware_temporal_portrait_identity_fusion",
                    requires_confirmation=True,
                ),
                **common_decision_fields,
            )

        if voice_likely or voice_uncertain:
            return build_snapshot(
                state="voice_session_candidate",
                matches_main_user=voice_likely,
                matched_user_id=self.config.primary_user_id,
                matched_user_display_name=normalized_voice_user_display_name,
                policy_recommendation="confirm_first",
                block_reason=(
                    "voice_only_requires_confirmation"
                    if voice_likely and not voice_spoof_uncertain
                    else "spoof_guard_uncertain"
                    if voice_spoof_uncertain
                    else "voice_signal_uncertain"
                ),
                claim=RuntimeClaimMetadata(
                    confidence=_weighted_confidence(
                        (normalized_voice_confidence, voice_reliability),
                        (session_support_ratio, 0.45),
                        (track_support_ratio, 0.35),
                    ),
                    source="quality_aware_voice_identity_fusion",
                    requires_confirmation=True,
                ),
                **common_decision_fields,
            )

        return build_snapshot(
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
            **common_decision_fields,
        )

    def _resolve_session_key(self, *, provided_session_key: int | None, observed_at: float) -> int | None:
        explicit_key = coerce_optional_int(provided_session_key)
        if explicit_key is not None:
            self._active_fallback_session_key = None
            self._active_fallback_session_seen_at = observed_at
            return explicit_key
        if (
            self._active_fallback_session_key is None
            or self._active_fallback_session_seen_at is None
            or observed_at - self._active_fallback_session_seen_at > self.config.session_reset_gap_s
        ):
            self._fallback_session_counter += 1
            self._active_fallback_session_key = -self._fallback_session_counter
        self._active_fallback_session_seen_at = observed_at
        return self._active_fallback_session_key

    def _append_history(
        self,
        *,
        observed_at: float,
        session_key: int | None,
        candidate_user_id: str | None,
        matches_main_user: bool,
        conflict: bool,
        visual_anchor_zone: str | None,
        visual_track_key: str | None,
        evidence_weight: float,
    ) -> None:
        entry = _IdentityFusionObservation(
            observed_at=observed_at,
            session_key=session_key,
            candidate_user_id=candidate_user_id,
            matches_main_user=matches_main_user,
            conflict=conflict,
            visual_anchor_zone=visual_anchor_zone,
            visual_track_key=visual_track_key,
            evidence_weight=_clip_ratio(evidence_weight),
        )
        if self._history:
            last = self._history[-1]
            if (
                last.session_key == entry.session_key
                and last.candidate_user_id == entry.candidate_user_id
                and last.matches_main_user == entry.matches_main_user
                and last.conflict == entry.conflict
                and last.visual_anchor_zone == entry.visual_anchor_zone
                and last.visual_track_key == entry.visual_track_key
                and observed_at - last.observed_at <= self.config.observation_dedup_window_s
            ):
                self._history[-1] = entry
                return
        self._history.append(entry)

    def _prune_history(self, *, observed_at: float) -> None:
        minimum_time = observed_at - self.config.temporal_window_s
        if not self._history:
            return
        retained = [
            item
            for item in self._history
            if item.observed_at >= minimum_time
        ]
        self._history = deque(retained, maxlen=self.config.max_observations)

    def _session_consistency(
        self,
        *,
        session_key: int | None,
        candidate_user_id: str | None,
        observed_at: float,
    ) -> tuple[int | None, float | None, int | None, str | None]:
        if candidate_user_id is None:
            return None, None, None, None
        same_session = [item for item in self._history if item.session_key == session_key]
        if not same_session:
            return None, None, None, None
        support = [item for item in same_session if item.candidate_user_id == candidate_user_id]
        if not support:
            return None, None, None, None
        total_weight = _decayed_weight_sum(
            same_session,
            observed_at=observed_at,
            half_life_s=self.config.evidence_half_life_s,
        )
        support_weight = _decayed_weight_sum(
            support,
            observed_at=observed_at,
            half_life_s=self.config.evidence_half_life_s,
        )
        conflict_count = sum(
            1
            for item in same_session
            if item.conflict or (item.candidate_user_id is not None and item.candidate_user_id != candidate_user_id)
        )
        support_ratio = round(support_weight / max(total_weight, 1e-6), 4)
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
        current_track_key: str | None,
        speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
        observed_at: float,
    ) -> tuple[int | None, float | None, str | None]:
        if speaker_association is not None and speaker_association.associated and speaker_association.confidence is not None:
            return 1, _clip_ratio(speaker_association.confidence), "speaker_locked"
        if candidate_user_id is None:
            return None, None, None
        same_session = [
            item
            for item in self._history
            if item.session_key == session_key and item.candidate_user_id == candidate_user_id
        ]
        if not same_session:
            return None, None, None

        if current_track_key:
            anchored = [item for item in same_session if item.visual_track_key]
            if not anchored:
                return None, None, None
            support = [item for item in anchored if item.visual_track_key == current_track_key]
            support_ratio = round(
                _decayed_weight_sum(support, observed_at=observed_at, half_life_s=self.config.evidence_half_life_s)
                / max(
                    _decayed_weight_sum(
                        anchored,
                        observed_at=observed_at,
                        half_life_s=self.config.evidence_half_life_s,
                    ),
                    1e-6,
                ),
                4,
            )
            if len(anchored) >= self.config.min_track_observations and support_ratio >= self.config.min_support_ratio:
                return len(anchored), support_ratio, "stable_track"
            if len(anchored) > 1:
                return len(anchored), support_ratio, "recent_track_shift"
            return len(anchored), support_ratio, "insufficient_history"

        if not current_zone:
            return None, None, None
        anchored = [item for item in same_session if item.visual_anchor_zone]
        if not anchored:
            return None, None, None
        support = [item for item in anchored if item.visual_anchor_zone == current_zone]
        support_ratio = round(
            _decayed_weight_sum(support, observed_at=observed_at, half_life_s=self.config.evidence_half_life_s)
            / max(
                _decayed_weight_sum(
                    anchored,
                    observed_at=observed_at,
                    half_life_s=self.config.evidence_half_life_s,
                ),
                1e-6,
            ),
            4,
        )
        if len(anchored) >= self.config.min_track_observations and support_ratio >= self.config.min_support_ratio:
            return len(anchored), support_ratio, "stable_anchor"
        if len(anchored) > 1:
            return len(anchored), support_ratio, "recent_anchor_shift"
        return len(anchored), support_ratio, "insufficient_history"


def _matched_user_id(portrait_match: PortraitMatchSnapshot | None) -> str | None:
    return None if portrait_match is None else _bounded_text(portrait_match.matched_user_id)


def _matched_user_display_name(portrait_match: PortraitMatchSnapshot | None) -> str | None:
    return None if portrait_match is None else _bounded_text(portrait_match.matched_user_display_name)


def _nested_mapping(value: Mapping[str, object] | None, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    child = value.get(key)
    return child if isinstance(child, Mapping) else {}


def _coalesce_value(mapping: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _camera_primary_person_zone(live_facts: Mapping[str, object] | None) -> str | None:
    camera = _nested_mapping(live_facts, "camera")
    value = normalize_text(camera.get("primary_person_zone")).lower()
    if value in {"left", "center", "right"}:
        return value
    return None


def _camera_primary_person_track_key(
    live_facts: Mapping[str, object] | None,
    *,
    speaker_association: ReSpeakerSpeakerAssociationSnapshot | None,
) -> str | None:
    camera = _nested_mapping(live_facts, "camera")
    for key in (
        "primary_person_track_id",
        "primary_person_track_key",
        "active_track_id",
        "track_id",
    ):
        candidate = _bounded_text(camera.get(key))
        if candidate:
            return candidate
    if speaker_association is not None:
        for attr in ("track_id", "track_key", "associated_track_id"):
            candidate = _bounded_text(getattr(speaker_association, attr, None))
            if candidate:
                return candidate
    return None


def _normalize_voice_status(value: object | None) -> str | None:
    normalized = normalize_text(value).lower()
    if not normalized:
        return None
    if normalized in _ALLOWED_VOICE_STATUSES:
        return normalized
    return "unknown_voice"


def _normalize_security_status(value: object | None) -> str | None:
    normalized = normalize_text(value).lower()
    if not normalized:
        return None
    aliases = {
        "bonafide": "bona_fide",
        "real": "live",
        "ok": "passed",
        "pass": "passed",
        "fail": "suspected_spoof",
        "likely_spoof": "suspected_spoof",
        "suspicious": "suspected_spoof",
        "tts": "deepfake",
        "vc": "deepfake",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in _ALLOWED_SECURITY_STATUSES:
        return normalized
    return None


def _security_posture(*, status: str | None, risk: float | None, fresh: bool) -> tuple[bool, bool]:
    if not fresh:
        return False, status in _BLOCKING_SECURITY_STATUSES or status in _SOFT_SECURITY_STATUSES or (risk is not None and risk >= 0.4)
    clipped_risk = _clip_ratio(risk)
    blocked = status in _BLOCKING_SECURITY_STATUSES or clipped_risk >= 0.7
    uncertain = not blocked and (
        status in _SOFT_SECURITY_STATUSES
        or (clipped_risk is not None and clipped_risk >= 0.4)
    )
    return blocked, uncertain


def _voice_reliability(
    *,
    status: str | None,
    confidence: float | None,
    fresh: bool,
    spoof_blocked: bool,
    spoof_uncertain: bool,
    spoof_status: str | None,
) -> float | None:
    if spoof_blocked:
        return 0.0
    if not fresh or status is None:
        return 0.0
    base = _clip_ratio(confidence) or 0.5
    if status == "likely_user":
        base = max(base, 0.72)
    elif status in {"uncertain", "uncertain_match"}:
        base = min(base, 0.45)
    elif status in {"ambiguous_match", "unknown_voice"}:
        base = min(base, 0.2)
    elif status == "known_other_user":
        base = max(base, 0.7)
    if spoof_uncertain or spoof_status not in _POSITIVE_SECURITY_STATUSES | {None}:
        base *= 0.65
    return round(_clip_ratio(base) or 0.0, 4)


def _portrait_reliability(
    *,
    portrait_state: str | None,
    portrait_confidence: float | None,
    face_spoof_blocked: bool,
    face_spoof_uncertain: bool,
    face_liveness_state: str | None,
) -> float | None:
    if face_spoof_blocked:
        return 0.0
    if portrait_state is None:
        return 0.0
    base = _clip_ratio(portrait_confidence) or 0.55
    if portrait_state in {"stable_match", "main_user_match", "known_other_user"}:
        base = max(base, 0.75)
    elif portrait_state == "uncertain_match":
        base = min(base, 0.5)
    elif portrait_state in {"unknown_face", "ambiguous_identity"}:
        base = min(base, 0.2)
    if face_spoof_uncertain or face_liveness_state not in _POSITIVE_SECURITY_STATUSES | {None}:
        base *= 0.7
    return round(_clip_ratio(base) or 0.0, 4)


def _temporal_reliability(
    *,
    session_support_ratio: float | None,
    track_support_ratio: float | None,
    temporal_state: str | None,
    speaker_association_confidence: float | None,
) -> float | None:
    values = [
        _clip_ratio(session_support_ratio),
        _clip_ratio(track_support_ratio),
        _clip_ratio(speaker_association_confidence),
    ]
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    base = sum(usable) / len(usable)
    if temporal_state == "stable_multimodal_match":
        base = max(base, 0.8)
    elif temporal_state == "recent_conflict":
        base = min(base, 0.25)
    return round(_clip_ratio(base) or 0.0, 4)


def _effective_event_time(observed_at: float | None, *, now_utc: datetime | None) -> float:
    effective_now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    now_ts = effective_now.timestamp()
    if observed_at is None:
        return round(now_ts, 3)
    try:
        parsed = float(observed_at)
    except (TypeError, ValueError):
        return round(now_ts, 3)
    if not isfinite(parsed):
        return round(now_ts, 3)
    if parsed > now_ts + _MAX_FUTURE_SKEW_S:
        return round(now_ts, 3)
    return round(parsed, 3)


def _checked_age_s(
    value: object | None,
    *,
    now_utc: datetime | None,
    fallback_observed_at: float | None = None,
) -> float | None:
    parsed = _parse_checked_at(value)
    effective_now = datetime.now(timezone.utc) if now_utc is None else now_utc.astimezone(timezone.utc)
    effective_now_ts = effective_now.timestamp()
    if parsed is None:
        if fallback_observed_at is None:
            return None
        if fallback_observed_at > effective_now_ts + _MAX_FUTURE_SKEW_S:
            return None
        return round(max(0.0, effective_now_ts - float(fallback_observed_at)), 3)
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


def _weighted_confidence(*pairs: tuple[float | None, float | None]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in pairs:
        clipped_value = _clip_ratio(value)
        clipped_weight = _clip_ratio(weight)
        if clipped_value is None or clipped_weight is None or clipped_weight <= 0.0:
            continue
        numerator += clipped_value * clipped_weight
        denominator += clipped_weight
    if denominator <= 0.0:
        return 0.0
    return round(max(0.0, min(1.0, numerator / denominator)), 4)


def _decayed_weight_sum(
    observations: list[_IdentityFusionObservation],
    *,
    observed_at: float,
    half_life_s: float,
) -> float:
    total = 0.0
    for item in observations:
        age_s = max(0.0, observed_at - item.observed_at)
        decay = 0.5 ** (age_s / max(half_life_s, 1e-6))
        total += max(0.05, item.evidence_weight) * decay
    return total


def _temporal_state(*, session_consistency_state: str | None, track_consistency_state: str | None) -> str | None:
    if session_consistency_state == "stable_session" and track_consistency_state in _STABLE_TRACK_STATES:
        return "stable_multimodal_match"
    if session_consistency_state == "recent_conflict" or track_consistency_state in _CONFLICT_TRACK_STATES:
        return "recent_conflict"
    if session_consistency_state is None and track_consistency_state is None:
        return None
    return "insufficient_history"


def _clip_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


def _bounded_text(value: object | None, *, max_len: int = _MAX_SAFE_TEXT_LEN) -> str | None:
    normalized = normalize_text(value)
    if not normalized:
        return None
    if len(normalized) > max_len:
        return normalized[: max_len - 1].rstrip() + "…"
    return normalized


def _coerce_positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not isfinite(parsed) or parsed <= 0.0:
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