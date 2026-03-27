"""Assess whether a wake candidate sounds like an enrolled household speaker."""

from __future__ import annotations

from dataclasses import dataclass

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.household_voice_identity import (
    HouseholdVoiceAssessment,
    HouseholdVoiceProfile,
    assess_household_voice_pcm16,
)


_FAMILIAR_SPEAKER_STATUSES = frozenset({"likely_user", "known_other_user"})


@dataclass(frozen=True, slots=True)
class FamiliarSpeakerWakeAssessment:
    """Describe whether a wake candidate matches an enrolled household voice."""

    assessment: HouseholdVoiceAssessment | None
    familiar: bool
    revision: str | None = None
    profile_count: int = 0

    def trace_details(self) -> dict[str, object]:
        assessment = self.assessment
        return {
            "familiar_speaker_revision": self.revision,
            "familiar_speaker_profile_count": self.profile_count,
            "familiar_speaker_status": None if assessment is None else assessment.status,
            "familiar_speaker_confidence": None if assessment is None else assessment.confidence,
            "familiar_speaker_user_id": None if assessment is None else assessment.matched_user_id,
            "familiar_speaker_user_display_name": (
                None if assessment is None else assessment.matched_user_display_name
            ),
            "familiar_speaker_match": self.familiar,
        }


def assess_familiar_speaker_pcm16(
    config: TwinrConfig,
    *,
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
    profiles: tuple[HouseholdVoiceProfile, ...],
    revision: str | None = None,
) -> FamiliarSpeakerWakeAssessment:
    """Score one wake candidate against the enrolled household speaker set."""

    if not profiles:
        return FamiliarSpeakerWakeAssessment(
            assessment=None,
            familiar=False,
            revision=revision,
            profile_count=0,
        )
    assessment = assess_household_voice_pcm16(
        pcm_bytes,
        sample_rate=sample_rate,
        channels=channels,
        checked_at=None,
        profiles=profiles,
        primary_user_id=getattr(config, "portrait_match_primary_user_id", "main_user") or "main_user",
        likely_threshold=getattr(config, "voice_profile_likely_threshold", 0.72),
        uncertain_threshold=getattr(config, "voice_profile_uncertain_threshold", 0.55),
        identity_margin=getattr(config, "household_voice_identity_margin", 0.06),
        min_sample_ms=getattr(config, "voice_profile_min_sample_ms", 1200),
    )
    familiar = (
        assessment.status in _FAMILIAR_SPEAKER_STATUSES
        and assessment.confidence is not None
        and assessment.confidence
        >= float(getattr(config, "voice_familiar_speaker_min_confidence", 0.82))
    )
    return FamiliarSpeakerWakeAssessment(
        assessment=assessment,
        familiar=familiar,
        revision=revision,
        profile_count=len(profiles),
    )


__all__ = [
    "FamiliarSpeakerWakeAssessment",
    "assess_familiar_speaker_pcm16",
]
