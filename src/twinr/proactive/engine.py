from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from twinr.agent.base_agent.config import TwinrConfig


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
        self._last_non_floor_pose_at: float | None = None
        self._possible_fall_candidate_at: float | None = None
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

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerEngine":
        return cls(user_name=config.user_display_name)

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

        candidates = [
            self._candidate_floor_stillness(now),
            self._candidate_possible_fall(now),
            self._candidate_distress_possible(now, vision),
            self._candidate_slumped_quiet(now),
            self._candidate_attention_window(now),
            self._candidate_showing_intent(now),
            self._candidate_positive_contact(now),
            self._candidate_person_returned(
                now,
                absence_duration=absence_duration,
                person_visible=vision.person_visible,
            ),
        ]
        ready = [candidate for candidate in candidates if candidate is not None]
        if not ready:
            return None
        decision = max(ready, key=lambda item: int(item.priority))
        self._last_triggered_at[decision.trigger_id] = now
        if decision.trigger_id in {"possible_fall", "floor_stillness"}:
            self._possible_fall_candidate_at = None
        return decision

    def _candidate_person_returned(
        self,
        now: float,
        *,
        absence_duration: float | None,
        person_visible: bool,
    ) -> SocialTriggerDecision | None:
        if absence_duration is None or not person_visible:
            return None
        if absence_duration < self.thresholds.person_returned_absence_s:
            return None
        if not self._recent_pir_motion(now):
            return None
        if self._cooldown_active("person_returned", now):
            return None
        return self._build_decision(
            trigger_id="person_returned",
            observed_at=now,
            priority=SocialTriggerPriority.PERSON_RETURNED,
            reason=f"Person visible again after {int(absence_duration)} seconds without presence.",
            prompt=self._with_name(
                base="Schön dich zu sehen. Wie geht's dir?",
                with_name="Hey {name}, schön dich zu sehen. Wie geht's dir?",
            ),
        )

    def _candidate_attention_window(self, now: float) -> SocialTriggerDecision | None:
        attention_since = self._conjunction_since(self._looking_since, self._quiet_since)
        if attention_since is None or (now - attention_since) < self.thresholds.attention_window_s:
            return None
        if self._cooldown_active("attention_window", now):
            return None
        return self._build_decision(
            trigger_id="attention_window",
            observed_at=now,
            priority=SocialTriggerPriority.ATTENTION_WINDOW,
            reason="Person was visible, looking toward the device, and quiet for a short attention window.",
            prompt="Kann ich dir bei etwas helfen?",
        )

    def _candidate_slumped_quiet(self, now: float) -> SocialTriggerDecision | None:
        active_since = self._conjunction_since(self._slumped_since, self._quiet_since, self._low_motion_since)
        if active_since is None or (now - active_since) < self.thresholds.slumped_quiet_s:
            return None
        if self._cooldown_active("slumped_quiet", now):
            return None
        return self._build_decision(
            trigger_id="slumped_quiet",
            observed_at=now,
            priority=SocialTriggerPriority.SLUMPED_QUIET,
            reason="Person stayed visibly slumped, quiet, and low-motion.",
            prompt=self._with_name(
                base="Ist alles in Ordnung?",
                with_name="Hey {name}, ist alles in Ordnung?",
            ),
        )

    def _candidate_possible_fall(self, now: float) -> SocialTriggerDecision | None:
        if self._possible_fall_candidate_at is None:
            return None
        if self._floor_since is None or self._low_motion_since is None:
            return None
        confirmation_since = self._conjunction_since(self._floor_since, self._low_motion_since)
        if confirmation_since is None:
            return None
        if (now - confirmation_since) < self.thresholds.possible_fall_stillness_s:
            return None
        if self._cooldown_active("possible_fall", now):
            return None
        return self._build_decision(
            trigger_id="possible_fall",
            observed_at=now,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
            reason="Body pose dropped to the floor shortly after an upright pose and then stayed low-motion.",
            prompt="Brauchst du Hilfe?",
        )

    def _candidate_floor_stillness(self, now: float) -> SocialTriggerDecision | None:
        active_since = self._conjunction_since(self._floor_since, self._quiet_since, self._low_motion_since)
        if active_since is None or (now - active_since) < self.thresholds.floor_stillness_s:
            return None
        if self._cooldown_active("floor_stillness", now):
            return None
        return self._build_decision(
            trigger_id="floor_stillness",
            observed_at=now,
            priority=SocialTriggerPriority.FLOOR_STILLNESS,
            reason="Person stayed low to the floor, quiet, and low-motion.",
            prompt=self._with_name(
                base="Antworte mir kurz: Ist alles okay?",
                with_name="Hey {name}, antworte mir kurz: Ist alles okay?",
            ),
        )

    def _candidate_showing_intent(self, now: float) -> SocialTriggerDecision | None:
        if self._showing_since is None or (now - self._showing_since) < self.thresholds.showing_intent_hold_s:
            return None
        if self._cooldown_active("showing_intent", now):
            return None
        return self._build_decision(
            trigger_id="showing_intent",
            observed_at=now,
            priority=SocialTriggerPriority.SHOWING_INTENT,
            reason="Person looked toward the device while holding a hand or object near the camera.",
            prompt="Möchtest du mir etwas zeigen?",
        )

    def _candidate_distress_possible(
        self,
        now: float,
        vision: SocialVisionObservation,
    ) -> SocialTriggerDecision | None:
        if self._distress_since is None or (now - self._distress_since) < self.thresholds.distress_hold_s:
            return None
        if not vision.person_visible and vision.body_pose != SocialBodyPose.SLUMPED:
            return None
        if self._cooldown_active("distress_possible", now):
            return None
        return self._build_decision(
            trigger_id="distress_possible",
            observed_at=now,
            priority=SocialTriggerPriority.DISTRESS_POSSIBLE,
            reason="Distress-like audio coincided with a visible or slumped person.",
            prompt=self._with_name(
                base="Ich wollte nur kurz fragen, ob alles in Ordnung ist.",
                with_name="Hey {name}, ich wollte nur kurz fragen, ob alles in Ordnung ist.",
            ),
        )

    def _candidate_positive_contact(self, now: float) -> SocialTriggerDecision | None:
        if self._smile_since is None or (now - self._smile_since) < self.thresholds.positive_contact_hold_s:
            return None
        if self._cooldown_active("positive_contact", now):
            return None
        return self._build_decision(
            trigger_id="positive_contact",
            observed_at=now,
            priority=SocialTriggerPriority.POSITIVE_CONTACT,
            reason="Person smiled while facing the device.",
            prompt=self._with_name(
                base="Schön, dich zu sehen. Was möchtest du machen?",
                with_name="Schön, dich zu sehen, {name}. Was möchtest du machen?",
            ),
        )

    def _update_presence_state(self, now: float, *, person_visible: bool) -> float | None:
        absence_duration: float | None = None
        if person_visible:
            if not self._person_visible and self._absence_started_at is not None:
                absence_duration = now - self._absence_started_at
            self._absence_started_at = None
        else:
            if self._absence_started_at is None:
                self._absence_started_at = now
        self._person_visible = person_visible
        return absence_duration

    def _update_pose_state(self, now: float, *, body_pose: SocialBodyPose) -> None:
        if body_pose in {SocialBodyPose.UPRIGHT, SocialBodyPose.SLUMPED}:
            self._last_non_floor_pose_at = now
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

    def _recent_pir_motion(self, now: float) -> bool:
        if self._last_pir_motion_at is None:
            return False
        return (now - self._last_pir_motion_at) <= self.thresholds.person_returned_recent_motion_s

    def _build_decision(
        self,
        *,
        trigger_id: str,
        observed_at: float,
        priority: SocialTriggerPriority,
        reason: str,
        prompt: str,
    ) -> SocialTriggerDecision:
        return SocialTriggerDecision(
            trigger_id=trigger_id,
            prompt=prompt,
            reason=reason,
            observed_at=observed_at,
            priority=priority,
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


__all__ = [
    "SocialAudioObservation",
    "SocialBodyPose",
    "SocialObservation",
    "SocialTriggerDecision",
    "SocialTriggerEngine",
    "SocialTriggerPriority",
    "SocialTriggerThresholds",
    "SocialVisionObservation",
]
