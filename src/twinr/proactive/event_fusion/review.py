"""Select review keyframes from recent fused-event evidence.

This module keeps keyframe-review planning separate from claim scoring. The
planner works on recent visual observations plus already-derived sequence
evidence and returns a compact onset/peak/latest plan that later review or
operator tooling can resolve into actual frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from twinr.proactive.event_fusion.audio_events import AudioMicroEvent
from twinr.proactive.event_fusion.buffers import RollingWindowBuffer, TimedSample
from twinr.proactive.event_fusion.vision_sequences import VisionSequenceEvent
from twinr.proactive.social.engine import SocialBodyPose, SocialMotionState, SocialVisionObservation


@dataclass(frozen=True, slots=True)
class ReviewKeyframeCandidate:
    """Describe one keyframe candidate chosen for downstream review."""

    observed_at: float
    role: str
    relevance_score: float
    person_visible: bool
    person_count: int
    body_pose: str
    motion_state: str
    looking_toward_device: bool
    smiling: bool

    def to_payload(self) -> dict[str, object]:
        """Serialize one candidate into plain review-plan facts."""

        return {
            "observed_at": self.observed_at,
            "role": self.role,
            "relevance_score": self.relevance_score,
            "person_visible": self.person_visible,
            "person_count": self.person_count,
            "body_pose": self.body_pose,
            "motion_state": self.motion_state,
            "looking_toward_device": self.looking_toward_device,
            "smiling": self.smiling,
        }


@dataclass(frozen=True, slots=True)
class KeyframeReviewPlan:
    """Describe a compact review plan for one fused event claim."""

    claim_state: str
    strategy: str = "relevance_plus_coverage"
    window_start_s: float | None = None
    window_end_s: float | None = None
    frames: tuple[ReviewKeyframeCandidate, ...] = field(default_factory=tuple)

    def to_payload(self) -> dict[str, object]:
        """Serialize the review plan into plain automation facts."""

        return {
            "claim_state": self.claim_state,
            "strategy": self.strategy,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "frames": [frame.to_payload() for frame in self.frames],
        }


@dataclass(frozen=True, slots=True)
class KeyframeReviewConfig:
    """Store bounded parameters for relevance-plus-coverage frame selection."""

    max_frames: int = 3
    pre_roll_s: float = 0.75
    audio_alignment_half_life_s: float = 1.5


def build_keyframe_review_plan(
    *,
    claim_state: str,
    observation_buffer: RollingWindowBuffer[SocialVisionObservation],
    vision_evidence: tuple[VisionSequenceEvent, ...],
    audio_evidence: tuple[AudioMicroEvent, ...],
    window_start_s: float | None,
    window_end_s: float | None,
    config: KeyframeReviewConfig | None = None,
) -> KeyframeReviewPlan | None:
    """Return a compact onset/peak/latest plan for one fused claim.

    The planner intentionally does not require actual frame images. It only
    picks the timestamps and local scene facts that a later review stage should
    resolve.
    """

    if window_start_s is None or window_end_s is None:
        return None
    cfg = config or KeyframeReviewConfig()
    candidate_samples = observation_buffer.between(max(0.0, window_start_s - cfg.pre_roll_s), window_end_s)
    if not candidate_samples:
        return None

    scored = [
        (
            sample,
            _candidate_score(
                sample=sample,
                now=window_end_s,
                vision_evidence=vision_evidence,
                audio_evidence=audio_evidence,
                config=cfg,
            ),
        )
        for sample in candidate_samples
    ]
    onset_sample = candidate_samples[0]
    peak_sample = max(scored, key=lambda item: (item[1], item[0].observed_at))[0]
    latest_sample = candidate_samples[-1]

    selected: list[ReviewKeyframeCandidate] = []
    used_timestamps: set[float] = set()
    for role, sample in (("onset", onset_sample), ("peak", peak_sample), ("latest", latest_sample)):
        if len(selected) >= max(1, cfg.max_frames):
            break
        if sample.observed_at in used_timestamps:
            continue
        used_timestamps.add(sample.observed_at)
        selected.append(_to_candidate(sample=sample, role=role, score_lookup=dict(scored)))

    return KeyframeReviewPlan(
        claim_state=claim_state,
        strategy="relevance_plus_coverage",
        window_start_s=window_start_s,
        window_end_s=window_end_s,
        frames=tuple(selected),
    )


def _to_candidate(
    *,
    sample: TimedSample[SocialVisionObservation],
    role: str,
    score_lookup: dict[TimedSample[SocialVisionObservation], float],
) -> ReviewKeyframeCandidate:
    """Convert one scored observation sample into a review candidate."""

    observation = sample.value
    return ReviewKeyframeCandidate(
        observed_at=sample.observed_at,
        role=role,
        relevance_score=round(score_lookup.get(sample, 0.0), 4),
        person_visible=observation.person_visible,
        person_count=observation.person_count,
        body_pose=_enum_value(observation.body_pose),
        motion_state=_enum_value(observation.motion_state),
        looking_toward_device=observation.looking_toward_device,
        smiling=observation.smiling,
    )


def _candidate_score(
    *,
    sample: TimedSample[SocialVisionObservation],
    now: float,
    vision_evidence: tuple[VisionSequenceEvent, ...],
    audio_evidence: tuple[AudioMicroEvent, ...],
    config: KeyframeReviewConfig,
) -> float:
    """Score one observation sample using evidence relevance plus scene salience."""

    observation = sample.value
    score = 0.0
    for sequence in vision_evidence:
        if sequence.window_start_s <= sample.observed_at <= sequence.window_end_s:
            score += sequence.confidence
        else:
            distance = min(
                abs(sample.observed_at - sequence.window_start_s),
                abs(sample.observed_at - sequence.window_end_s),
            )
            score += sequence.confidence * _exp_half_life(distance, half_life_s=1.25)
    for event in audio_evidence:
        score += event.confidence * _exp_half_life(
            abs(sample.observed_at - event.observed_at),
            half_life_s=config.audio_alignment_half_life_s,
        ) * 0.5
    if observation.person_visible:
        score += 0.12
    if _enum_value(observation.body_pose) in {SocialBodyPose.FLOOR.value, SocialBodyPose.LYING_LOW.value}:
        score += 0.2
    elif _enum_value(observation.body_pose) == SocialBodyPose.SLUMPED.value:
        score += 0.14
    if observation.smiling and (observation.looking_toward_device or observation.engaged_with_device):
        score += 0.14
    if observation.hand_or_object_near_camera or observation.showing_intent_likely:
        score += 0.12
    if _enum_value(observation.motion_state) == SocialMotionState.STILL.value:
        score += 0.05
    score += 0.05 * _exp_half_life(max(0.0, now - sample.observed_at), half_life_s=2.0)
    return round(score, 4)


def _exp_half_life(age_s: float, *, half_life_s: float) -> float:
    """Return an exponential decay weight for one age/half-life pair."""

    if half_life_s <= 0.0:
        return 0.0
    return math.exp(-math.log(2.0) * max(0.0, age_s) / half_life_s)


def _enum_value(value: object | None) -> str:
    """Return the string value for one enum-like field."""

    return getattr(value, "value", value) if value is not None else "unknown"


__all__ = [
    "KeyframeReviewConfig",
    "KeyframeReviewPlan",
    "ReviewKeyframeCandidate",
    "build_keyframe_review_plan",
]
