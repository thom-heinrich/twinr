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
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot


_DEFAULT_VOICE_ASSESSMENT_MAX_AGE_S = 120
_MAX_FUTURE_SKEW_S = 5.0
_ALLOWED_VOICE_STATUSES = frozenset({"likely_user", "uncertain", "unknown_voice"})


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
        )
    if not normalized_status:
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
        )
    if checked_age_s is None or checked_age_s > max(1, int(max_voice_age_s)):
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
        )
    if normalized_status == "likely_user":
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
        )
    if normalized_status == "uncertain":
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
    )


def _normalize_voice_status(value: object | None) -> str:
    """Return one normalized voice-status token or an empty string."""

    normalized = normalize_text(value).lower()
    if normalized in _ALLOWED_VOICE_STATUSES:
        return normalized
    return ""


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
