"""Define fused event claims and their fail-closed delivery gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from twinr.proactive.social.engine import SocialObservation
from twinr.proactive.event_fusion.review import KeyframeReviewPlan


class FusionActionLevel(StrEnum):
    """Describe how far one fused event may travel in runtime policy."""

    IGNORE = "ignore"
    DIRECT = "direct"
    PROMPT_ONLY = "prompt_only"
    REVIEW_ONLY = "review_only"


@dataclass(frozen=True, slots=True)
class EventFusionPolicyContext:
    """Capture the small V1 gating context used by fused claims."""

    background_media_likely: bool = False
    room_busy_or_overlapping: bool = False
    multi_person_context: bool = False

    @classmethod
    def from_observation(
        cls,
        observation: SocialObservation,
        *,
        room_busy_or_overlapping: bool = False,
    ) -> "EventFusionPolicyContext":
        """Build the V1 gate context from one normalized social observation."""

        multi_person = observation.inspected and observation.vision.person_visible and observation.vision.person_count > 1
        return cls(
            background_media_likely=observation.audio.background_media_likely is True,
            room_busy_or_overlapping=(
                room_busy_or_overlapping
                or observation.audio.speech_overlap_likely is True
            ),
            multi_person_context=multi_person,
        )

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        """Return the active V1 block reasons in stable order."""

        reasons: list[str] = []
        if self.background_media_likely:
            reasons.append("background_media_active")
        if self.room_busy_or_overlapping:
            reasons.append("room_busy_or_overlapping")
        if self.multi_person_context:
            reasons.append("multi_person_context")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class FusedEventClaim:
    """Describe one short-window multimodal event claim."""

    state: str
    active: bool
    confidence: float
    source: str
    source_type: str = "observed"
    requires_confirmation: bool = True
    window_start_s: float | None = None
    window_end_s: float | None = None
    action_level: FusionActionLevel = FusionActionLevel.IGNORE
    delivery_allowed: bool = False
    blocked_by: tuple[str, ...] = field(default_factory=tuple)
    supporting_audio_events: tuple[str, ...] = field(default_factory=tuple)
    supporting_vision_events: tuple[str, ...] = field(default_factory=tuple)
    review_recommended: bool = False
    keyframe_review_plan: KeyframeReviewPlan | None = None

    def to_payload(self) -> dict[str, object]:
        """Serialize one fused claim into plain automation facts."""

        return {
            "state": self.state,
            "active": self.active,
            "confidence": self.confidence,
            "source": self.source,
            "source_type": self.source_type,
            "requires_confirmation": self.requires_confirmation,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "action_level": self.action_level.value,
            "delivery_allowed": self.delivery_allowed,
            "blocked_by": list(self.blocked_by),
            "supporting_audio_events": list(self.supporting_audio_events),
            "supporting_vision_events": list(self.supporting_vision_events),
            "review_recommended": self.review_recommended,
            "keyframe_review_plan": (
                None
                if self.keyframe_review_plan is None
                else self.keyframe_review_plan.to_payload()
            ),
        }


def build_fused_claim(
    *,
    state: str,
    confidence: float,
    source: str,
    policy_context: EventFusionPolicyContext,
    window_start_s: float | None,
    window_end_s: float | None,
    preferred_action_level: FusionActionLevel,
    supporting_audio_events: tuple[str, ...] = (),
    supporting_vision_events: tuple[str, ...] = (),
    requires_confirmation: bool = True,
    review_recommended: bool = False,
    keyframe_review_plan: KeyframeReviewPlan | None = None,
) -> FusedEventClaim:
    """Build one fused claim while applying the V1 fail-closed gates."""

    blocked_by = policy_context.blocked_reasons
    delivery_allowed = not blocked_by
    action_level = preferred_action_level if delivery_allowed else FusionActionLevel.IGNORE
    return FusedEventClaim(
        state=state,
        active=True,
        confidence=confidence,
        source=source,
        source_type="observed_fused",
        requires_confirmation=True if preferred_action_level is not FusionActionLevel.DIRECT else requires_confirmation,
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        action_level=action_level,
        delivery_allowed=delivery_allowed,
        blocked_by=blocked_by,
        supporting_audio_events=supporting_audio_events,
        supporting_vision_events=supporting_vision_events,
        review_recommended=review_recommended,
        keyframe_review_plan=keyframe_review_plan,
    )


__all__ = [
    "EventFusionPolicyContext",
    "FusedEventClaim",
    "FusionActionLevel",
    "build_fused_claim",
]
