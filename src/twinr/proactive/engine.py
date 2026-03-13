from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.scoring import TriggerScoreEvidence, bool_score, hold_progress, recent_progress, weighted_trigger_score


class SocialBodyPose(StrEnum):
    UNKNOWN = "unknown"
    UPRIGHT = "upright"
    SLUMPED = "slumped"
    FLOOR = "floor"


class SocialTriggerPriority(IntEnum):
    POSITIVE_CONTACT = 10
    PERSON_RETURNED = 20
    SHOWING_INTENT = 30
    ATTENTION_WINDOW = 40
    SLUMPED_QUIET = 60
    DISTRESS_POSSIBLE = 70
    POSSIBLE_FALL = 80
    FLOOR_STILLNESS = 90


@dataclass(frozen=True, slots=True)
class SocialVisionObservation:
    person_visible: bool = False
    looking_toward_device: bool = False
    body_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
    smiling: bool = False
    hand_or_object_near_camera: bool = False


@dataclass(frozen=True, slots=True)
class SocialAudioObservation:
    speech_detected: bool | None = None
    distress_detected: bool | None = None


@dataclass(frozen=True, slots=True)
class SocialObservation:
    observed_at: float
    pir_motion_detected: bool = False
    low_motion: bool = False
    vision: SocialVisionObservation = field(default_factory=SocialVisionObservation)
    audio: SocialAudioObservation = field(default_factory=SocialAudioObservation)


@dataclass(frozen=True, slots=True)
class SocialTriggerDecision:
    trigger_id: str
    prompt: str
    reason: str
    observed_at: float
    priority: SocialTriggerPriority
    score: float = 1.0
    threshold: float = 1.0
    evidence: tuple[TriggerScoreEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class SocialTriggerEvaluation:
    trigger_id: str
    prompt: str
    reason: str
    observed_at: float
    priority: SocialTriggerPriority
    score: float
    threshold: float
    evidence: tuple[TriggerScoreEvidence, ...] = ()
    passed: bool = False
    blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SocialTriggerThresholds:
    person_returned_absence_s: float = 20.0 * 60.0
    person_returned_recent_motion_s: float = 30.0
    attention_window_s: float = 6.0
    slumped_quiet_s: float = 20.0
    possible_fall_stillness_s: float = 10.0
    floor_stillness_s: float = 20.0
    showing_intent_hold_s: float = 1.5
    positive_contact_hold_s: float = 1.5
    distress_hold_s: float = 3.0
    fall_transition_window_s: float = 8.0
    person_returned_score_threshold: float = 0.9
    attention_window_score_threshold: float = 0.86
    slumped_quiet_score_threshold: float = 0.9
    possible_fall_score_threshold: float = 0.82
    floor_stillness_score_threshold: float = 0.9
    showing_intent_score_threshold: float = 0.84
    positive_contact_score_threshold: float = 0.84
    distress_possible_score_threshold: float = 0.85

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerThresholds":
        return cls(
            person_returned_absence_s=config.proactive_person_returned_absence_s,
            person_returned_recent_motion_s=config.proactive_person_returned_recent_motion_s,
            attention_window_s=config.proactive_attention_window_s,
            slumped_quiet_s=config.proactive_slumped_quiet_s,
            possible_fall_stillness_s=config.proactive_possible_fall_stillness_s,
            floor_stillness_s=config.proactive_floor_stillness_s,
            showing_intent_hold_s=config.proactive_showing_intent_hold_s,
            positive_contact_hold_s=config.proactive_positive_contact_hold_s,
            distress_hold_s=config.proactive_distress_hold_s,
            fall_transition_window_s=config.proactive_fall_transition_window_s,
            person_returned_score_threshold=config.proactive_person_returned_score_threshold,
            attention_window_score_threshold=config.proactive_attention_window_score_threshold,
            slumped_quiet_score_threshold=config.proactive_slumped_quiet_score_threshold,
            possible_fall_score_threshold=config.proactive_possible_fall_score_threshold,
            floor_stillness_score_threshold=config.proactive_floor_stillness_score_threshold,
            showing_intent_score_threshold=config.proactive_showing_intent_score_threshold,
            positive_contact_score_threshold=config.proactive_positive_contact_score_threshold,
            distress_possible_score_threshold=config.proactive_distress_possible_score_threshold,
        )


class SocialTriggerEngine:
    def __init__(
        self,
        *,
        user_name: str | None = None,
        thresholds: SocialTriggerThresholds | None = None,
    ) -> None:
        self.user_name = (user_name or "").strip() or None
        self.thresholds = thresholds or SocialTriggerThresholds()
        self._cooldowns: dict[str, float] = {
            "person_returned": 30.0 * 60.0,
            "attention_window": 10.0 * 60.0,
            "slumped_quiet": 20.0 * 60.0,
            "possible_fall": 60.0,
            "floor_stillness": 60.0,
            "showing_intent": 5.0 * 60.0,
            "distress_possible": 15.0 * 60.0,
            "positive_contact": 20.0 * 60.0,
        }
        self._last_triggered_at: dict[str, float] = {}
        self._last_pir_motion_at: float | None = None
        self._absence_started_at: float | None = None
        self._visible_since: float | None = None
        self._last_non_floor_pose_at: float | None = None
        self._last_slumped_at: float | None = None
        self._possible_fall_candidate_at: float | None = None
        self._possible_fall_loss_candidate_at: float | None = None
        self._possible_fall_loss_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
        self._possible_fall_loss_visible_duration_s: float | None = None
        self._person_visible: bool = False
        self._current_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
        self._quiet_since: float | None = None
        self._looking_since: float | None = None
        self._slumped_since: float | None = None
        self._floor_since: float | None = None
        self._low_motion_since: float | None = None
        self._showing_since: float | None = None
        self._smile_since: float | None = None
        self._distress_since: float | None = None
        self._last_evaluations: tuple[SocialTriggerEvaluation, ...] = ()

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerEngine":
        return cls(
            user_name=config.user_display_name,
            thresholds=SocialTriggerThresholds.from_config(config),
        )

    @property
    def last_evaluations(self) -> tuple[SocialTriggerEvaluation, ...]:
        return self._last_evaluations

    @property
    def best_evaluation(self) -> SocialTriggerEvaluation | None:
        if not self._last_evaluations:
            return None
        return max(
            self._last_evaluations,
            key=lambda item: (item.score, int(item.priority)),
        )

    def observe(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        now = observation.observed_at
        vision = observation.vision
        audio = observation.audio

        if observation.pir_motion_detected:
            self._last_pir_motion_at = now

        absence_duration = self._update_presence_state(now, person_visible=vision.person_visible)
        self._update_pose_state(now, body_pose=vision.body_pose)
        self._quiet_since = self._next_since(audio.speech_detected is False, self._quiet_since, now)
        self._looking_since = self._next_since(
            vision.person_visible and vision.looking_toward_device,
            self._looking_since,
            now,
        )
        self._slumped_since = self._next_since(
            vision.person_visible and vision.body_pose == SocialBodyPose.SLUMPED,
            self._slumped_since,
            now,
        )
        self._floor_since = self._next_since(
            vision.person_visible and vision.body_pose == SocialBodyPose.FLOOR,
            self._floor_since,
            now,
        )
        self._low_motion_since = self._next_since(observation.low_motion, self._low_motion_since, now)
        self._showing_since = self._next_since(
            vision.person_visible and vision.looking_toward_device and vision.hand_or_object_near_camera,
            self._showing_since,
            now,
        )
        self._smile_since = self._next_since(
            vision.person_visible and vision.looking_toward_device and vision.smiling,
            self._smile_since,
            now,
        )
        self._distress_since = self._next_since(audio.distress_detected is True, self._distress_since, now)

        evaluations = (
            self._candidate_floor_stillness(now),
            self._candidate_possible_fall(now),
            self._candidate_distress_possible(now, vision),
            self._candidate_slumped_quiet(now),
            self._candidate_attention_window(now),
            self._candidate_showing_intent(now, vision),
            self._candidate_positive_contact(now, vision),
            self._candidate_person_returned(
                now,
                absence_duration=absence_duration,
                person_visible=vision.person_visible,
            ),
        )
        self._last_evaluations = evaluations

        ready = [candidate for candidate in evaluations if candidate.passed]
        if not ready:
            return None
        selected = max(
            ready,
            key=lambda item: (int(item.priority), item.score),
        )
        self._last_triggered_at[selected.trigger_id] = now
        if selected.trigger_id in {"possible_fall", "floor_stillness"}:
            self._possible_fall_candidate_at = None
            self._possible_fall_loss_candidate_at = None
            self._possible_fall_loss_pose = SocialBodyPose.UNKNOWN
            self._possible_fall_loss_visible_duration_s = None
        return SocialTriggerDecision(
            trigger_id=selected.trigger_id,
            prompt=selected.prompt,
            reason=selected.reason,
            observed_at=selected.observed_at,
            priority=selected.priority,
            score=selected.score,
            threshold=selected.threshold,
            evidence=selected.evidence,
        )

    def _candidate_person_returned(
        self,
        now: float,
        *,
        absence_duration: float | None,
        person_visible: bool,
    ) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="absence_hold",
                value=0.0 if absence_duration is None else min(1.0, absence_duration / self.thresholds.person_returned_absence_s),
                weight=0.4,
                detail=(
                    f"absence={self._seconds(absence_duration)} target={self.thresholds.person_returned_absence_s:.1f}s"
                    if absence_duration is not None
                    else "no completed absence window"
                ),
            ),
            TriggerScoreEvidence(
                key="recent_pir_motion",
                value=recent_progress(now, self._last_pir_motion_at, self.thresholds.person_returned_recent_motion_s),
                weight=0.4,
                detail=(
                    f"last_motion_age={self._seconds(None if self._last_pir_motion_at is None else now - self._last_pir_motion_at)}"
                ),
            ),
            TriggerScoreEvidence(
                key="person_visible",
                value=bool_score(person_visible),
                weight=0.2,
                detail=f"person_visible={person_visible}",
            ),
        )
        blocked_reason = None
        if absence_duration is None:
            blocked_reason = "no_completed_absence_window"
        elif self._cooldown_active("person_returned", now):
            blocked_reason = "cooldown_active"
        return self._evaluate_candidate(
            trigger_id="person_returned",
            observed_at=now,
            priority=SocialTriggerPriority.PERSON_RETURNED,
            threshold=self.thresholds.person_returned_score_threshold,
            reason=(
                f"Person visible again after {int(absence_duration)} seconds without presence."
                if absence_duration is not None
                else "Person has not yet completed a tracked absence window."
            ),
            prompt=self._with_name(
                base="Schön dich zu sehen. Wie geht's dir?",
                with_name="Hey {name}, schön dich zu sehen. Wie geht's dir?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_attention_window(self, now: float) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="looking_hold",
                value=hold_progress(now, self._looking_since, self.thresholds.attention_window_s),
                weight=0.45,
                detail=self._hold_detail(self._looking_since, now, self.thresholds.attention_window_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.attention_window_s),
                weight=0.45,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.attention_window_s),
            ),
            TriggerScoreEvidence(
                key="co_present_attention",
                value=bool_score(self._conjunction_since(self._looking_since, self._quiet_since) is not None),
                weight=0.1,
                detail="looking and quiet overlap",
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("attention_window", now) else None
        return self._evaluate_candidate(
            trigger_id="attention_window",
            observed_at=now,
            priority=SocialTriggerPriority.ATTENTION_WINDOW,
            threshold=self.thresholds.attention_window_score_threshold,
            reason="Person was visible, looking toward the device, and quiet for a short attention window.",
            prompt="Kann ich dir bei etwas helfen?",
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_slumped_quiet(self, now: float) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="slumped_hold",
                value=hold_progress(now, self._slumped_since, self.thresholds.slumped_quiet_s),
                weight=0.35,
                detail=self._hold_detail(self._slumped_since, now, self.thresholds.slumped_quiet_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.slumped_quiet_s),
                weight=0.3,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.slumped_quiet_s),
            ),
            TriggerScoreEvidence(
                key="low_motion_hold",
                value=hold_progress(now, self._low_motion_since, self.thresholds.slumped_quiet_s),
                weight=0.25,
                detail=self._hold_detail(self._low_motion_since, now, self.thresholds.slumped_quiet_s),
            ),
            TriggerScoreEvidence(
                key="concurrent_signals",
                value=bool_score(
                    self._conjunction_since(
                        self._slumped_since,
                        self._quiet_since,
                        self._low_motion_since,
                    )
                    is not None
                ),
                weight=0.1,
                detail="slumped, quiet, and low-motion overlap",
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("slumped_quiet", now) else None
        return self._evaluate_candidate(
            trigger_id="slumped_quiet",
            observed_at=now,
            priority=SocialTriggerPriority.SLUMPED_QUIET,
            threshold=self.thresholds.slumped_quiet_score_threshold,
            reason="Person stayed visibly slumped, quiet, and low-motion.",
            prompt=self._with_name(
                base="Ist alles in Ordnung?",
                with_name="Hey {name}, ist alles in Ordnung?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_possible_fall(self, now: float) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="fall_transition_signal",
                value=self._possible_fall_transition_signal(),
                weight=0.41,
                detail=self._possible_fall_transition_detail(now),
            ),
            TriggerScoreEvidence(
                key="low_or_missing_hold",
                value=self._possible_fall_low_or_missing_hold(now),
                weight=0.28,
                detail=self._possible_fall_low_or_missing_detail(now),
            ),
            TriggerScoreEvidence(
                key="low_motion_hold",
                value=hold_progress(now, self._low_motion_since, self.thresholds.possible_fall_stillness_s),
                weight=0.28,
                detail=self._hold_detail(self._low_motion_since, now, self.thresholds.possible_fall_stillness_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.possible_fall_stillness_s),
                weight=0.15,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.possible_fall_stillness_s),
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("possible_fall", now) else None
        return self._evaluate_candidate(
            trigger_id="possible_fall",
            observed_at=now,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
            threshold=self.thresholds.possible_fall_score_threshold,
            reason="Person dropped sharply lower or disappeared from view after being visible and then stayed still.",
            prompt="Brauchst du Hilfe?",
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_floor_stillness(self, now: float) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="floor_hold",
                value=hold_progress(now, self._floor_since, self.thresholds.floor_stillness_s),
                weight=0.38,
                detail=self._hold_detail(self._floor_since, now, self.thresholds.floor_stillness_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, self._quiet_since, self.thresholds.floor_stillness_s),
                weight=0.32,
                detail=self._hold_detail(self._quiet_since, now, self.thresholds.floor_stillness_s),
            ),
            TriggerScoreEvidence(
                key="low_motion_hold",
                value=hold_progress(now, self._low_motion_since, self.thresholds.floor_stillness_s),
                weight=0.2,
                detail=self._hold_detail(self._low_motion_since, now, self.thresholds.floor_stillness_s),
            ),
            TriggerScoreEvidence(
                key="concurrent_signals",
                value=bool_score(
                    self._conjunction_since(
                        self._floor_since,
                        self._quiet_since,
                        self._low_motion_since,
                    )
                    is not None
                ),
                weight=0.1,
                detail="floor, quiet, and low-motion overlap",
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("floor_stillness", now) else None
        return self._evaluate_candidate(
            trigger_id="floor_stillness",
            observed_at=now,
            priority=SocialTriggerPriority.FLOOR_STILLNESS,
            threshold=self.thresholds.floor_stillness_score_threshold,
            reason="Person stayed low to the floor, quiet, and low-motion.",
            prompt=self._with_name(
                base="Antworte mir kurz: Ist alles okay?",
                with_name="Hey {name}, antworte mir kurz: Ist alles okay?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_showing_intent(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="showing_hold",
                value=hold_progress(now, self._showing_since, self.thresholds.showing_intent_hold_s),
                weight=0.55,
                detail=self._hold_detail(self._showing_since, now, self.thresholds.showing_intent_hold_s),
            ),
            TriggerScoreEvidence(
                key="hand_or_object_near_camera",
                value=bool_score(vision.hand_or_object_near_camera),
                weight=0.25,
                detail=f"hand_or_object_near_camera={vision.hand_or_object_near_camera}",
            ),
            TriggerScoreEvidence(
                key="looking_toward_device",
                value=bool_score(vision.looking_toward_device),
                weight=0.2,
                detail=f"looking_toward_device={vision.looking_toward_device}",
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("showing_intent", now) else None
        return self._evaluate_candidate(
            trigger_id="showing_intent",
            observed_at=now,
            priority=SocialTriggerPriority.SHOWING_INTENT,
            threshold=self.thresholds.showing_intent_score_threshold,
            reason="Person looked toward the device while holding a hand or object near the camera.",
            prompt="Möchtest du mir etwas zeigen?",
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_distress_possible(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="distress_hold",
                value=hold_progress(now, self._distress_since, self.thresholds.distress_hold_s),
                weight=0.65,
                detail=self._hold_detail(self._distress_since, now, self.thresholds.distress_hold_s),
            ),
            TriggerScoreEvidence(
                key="visible_or_slumped_person",
                value=bool_score(vision.person_visible or vision.body_pose == SocialBodyPose.SLUMPED),
                weight=0.35,
                detail=f"person_visible={vision.person_visible} body_pose={vision.body_pose.value}",
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("distress_possible", now) else None
        return self._evaluate_candidate(
            trigger_id="distress_possible",
            observed_at=now,
            priority=SocialTriggerPriority.DISTRESS_POSSIBLE,
            threshold=self.thresholds.distress_possible_score_threshold,
            reason="Distress-like audio coincided with a visible or slumped person.",
            prompt=self._with_name(
                base="Ich wollte nur kurz fragen, ob alles in Ordnung ist.",
                with_name="Hey {name}, ich wollte nur kurz fragen, ob alles in Ordnung ist.",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _candidate_positive_contact(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerEvaluation:
        evidence = (
            TriggerScoreEvidence(
                key="smile_hold",
                value=hold_progress(now, self._smile_since, self.thresholds.positive_contact_hold_s),
                weight=0.55,
                detail=self._hold_detail(self._smile_since, now, self.thresholds.positive_contact_hold_s),
            ),
            TriggerScoreEvidence(
                key="person_visible",
                value=bool_score(vision.person_visible),
                weight=0.15,
                detail=f"person_visible={vision.person_visible}",
            ),
            TriggerScoreEvidence(
                key="looking_toward_device",
                value=bool_score(vision.looking_toward_device),
                weight=0.15,
                detail=f"looking_toward_device={vision.looking_toward_device}",
            ),
            TriggerScoreEvidence(
                key="smiling_now",
                value=bool_score(vision.smiling),
                weight=0.15,
                detail=f"smiling={vision.smiling}",
            ),
        )
        blocked_reason = "cooldown_active" if self._cooldown_active("positive_contact", now) else None
        return self._evaluate_candidate(
            trigger_id="positive_contact",
            observed_at=now,
            priority=SocialTriggerPriority.POSITIVE_CONTACT,
            threshold=self.thresholds.positive_contact_score_threshold,
            reason="Person smiled while facing the device.",
            prompt=self._with_name(
                base="Schön, dich zu sehen. Was möchtest du machen?",
                with_name="Schön, dich zu sehen, {name}. Was möchtest du machen?",
            ),
            evidence=evidence,
            blocked_reason=blocked_reason,
        )

    def _update_presence_state(self, now: float, *, person_visible: bool) -> float | None:
        absence_duration: float | None = None
        if person_visible:
            if not self._person_visible and self._absence_started_at is not None:
                absence_duration = now - self._absence_started_at
            self._absence_started_at = None
            self._possible_fall_loss_candidate_at = None
            self._possible_fall_loss_pose = SocialBodyPose.UNKNOWN
            self._possible_fall_loss_visible_duration_s = None
            if self._visible_since is None:
                self._visible_since = now
        else:
            if self._person_visible:
                visible_duration = None if self._visible_since is None else now - self._visible_since
                if self._fell_out_of_view_after_fall_like_presence(now, visible_duration=visible_duration):
                    self._possible_fall_loss_candidate_at = now
                    self._possible_fall_loss_pose = self._current_pose
                    self._possible_fall_loss_visible_duration_s = visible_duration
                self._visible_since = None
            if self._absence_started_at is None:
                self._absence_started_at = now
        self._person_visible = person_visible
        return absence_duration

    def _update_pose_state(self, now: float, *, body_pose: SocialBodyPose) -> None:
        if body_pose in {SocialBodyPose.UPRIGHT, SocialBodyPose.SLUMPED}:
            self._last_non_floor_pose_at = now
        if body_pose == SocialBodyPose.SLUMPED:
            self._last_slumped_at = now
        if body_pose == self._current_pose:
            return
        self._current_pose = body_pose
        if body_pose == SocialBodyPose.FLOOR:
            if (
                self._last_non_floor_pose_at is not None
                and (now - self._last_non_floor_pose_at) <= self.thresholds.fall_transition_window_s
            ):
                self._possible_fall_candidate_at = now
        else:
            self._possible_fall_candidate_at = None

    def _cooldown_active(self, trigger_id: str, now: float) -> bool:
        last_at = self._last_triggered_at.get(trigger_id)
        if last_at is None:
            return False
        return (now - last_at) < self._cooldowns[trigger_id]

    def _evaluate_candidate(
        self,
        *,
        trigger_id: str,
        observed_at: float,
        priority: SocialTriggerPriority,
        threshold: float,
        reason: str,
        prompt: str,
        evidence: tuple[TriggerScoreEvidence, ...],
        blocked_reason: str | None = None,
    ) -> SocialTriggerEvaluation:
        score_card = weighted_trigger_score(
            threshold=threshold,
            evidence=evidence,
        )
        return SocialTriggerEvaluation(
            trigger_id=trigger_id,
            prompt=prompt,
            reason=reason,
            observed_at=observed_at,
            priority=priority,
            score=score_card.score,
            threshold=score_card.threshold,
            evidence=score_card.evidence,
            passed=(blocked_reason is None and score_card.passed),
            blocked_reason=blocked_reason,
        )

    def _with_name(self, *, base: str, with_name: str) -> str:
        if self.user_name is None:
            return base
        return with_name.format(name=self.user_name)

    def _conjunction_since(self, *starts: float | None) -> float | None:
        active_starts = [start for start in starts if start is not None]
        if len(active_starts) != len(starts):
            return None
        return max(active_starts)

    def _next_since(self, active: bool, current: float | None, now: float) -> float | None:
        if not active:
            return None
        if current is None:
            return now
        return current

    def _hold_detail(self, since: float | None, now: float, target_s: float) -> str:
        if since is None:
            return f"active_for=0.0s target={target_s:.1f}s"
        return f"active_for={now - since:.1f}s target={target_s:.1f}s"

    def _fell_out_of_view_after_fall_like_presence(self, now: float, *, visible_duration: float | None) -> bool:
        if self._possible_fall_candidate_at is not None:
            return True
        if visible_duration is None or visible_duration < self._fall_visibility_loss_arming_s():
            return False
        if self._current_pose == SocialBodyPose.SLUMPED and self._last_slumped_at is not None:
            return (now - self._last_slumped_at) <= self.thresholds.fall_transition_window_s
        if self._current_pose == SocialBodyPose.UPRIGHT and self._last_non_floor_pose_at is not None:
            return (now - self._last_non_floor_pose_at) <= self.thresholds.fall_transition_window_s
        return False

    def _possible_fall_transition_signal(self) -> float:
        if self._possible_fall_candidate_at is not None:
            return 1.0
        if self._possible_fall_loss_candidate_at is None:
            return 0.0
        if self._possible_fall_loss_pose == SocialBodyPose.SLUMPED:
            return 0.9
        if self._possible_fall_loss_pose == SocialBodyPose.UPRIGHT:
            return 0.75
        return 0.55

    def _possible_fall_low_or_missing_hold(self, now: float) -> float:
        floor_hold = hold_progress(now, self._floor_since, self.thresholds.possible_fall_stillness_s)
        missing_hold = hold_progress(now, self._possible_fall_loss_candidate_at, self.thresholds.possible_fall_stillness_s)
        return max(floor_hold, missing_hold)

    def _possible_fall_low_or_missing_detail(self, now: float) -> str:
        floor_detail = self._hold_detail(self._floor_since, now, self.thresholds.possible_fall_stillness_s)
        missing_detail = self._hold_detail(
            self._possible_fall_loss_candidate_at,
            now,
            self.thresholds.possible_fall_stillness_s,
        )
        return f"floor={floor_detail}; missing={missing_detail}"

    def _possible_fall_transition_detail(self, now: float) -> str:
        if self._possible_fall_candidate_at is not None:
            return f"upright_to_floor_age={self._seconds(now - self._possible_fall_candidate_at)}"
        if self._possible_fall_loss_candidate_at is not None:
            return (
                f"{self._possible_fall_loss_pose.value}_to_visibility_loss_age="
                f"{self._seconds(now - self._possible_fall_loss_candidate_at)} "
                f"visible_for={self._seconds(self._possible_fall_loss_visible_duration_s)}"
            )
        return "no recent fall-like transition"

    def _fall_visibility_loss_arming_s(self) -> float:
        return 2.0

    def _seconds(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{value:.1f}s"


__all__ = [
    "SocialAudioObservation",
    "SocialBodyPose",
    "SocialObservation",
    "SocialTriggerDecision",
    "SocialTriggerEngine",
    "SocialTriggerEvaluation",
    "SocialTriggerPriority",
    "SocialTriggerThresholds",
    "SocialVisionObservation",
]
