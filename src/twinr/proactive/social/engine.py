"""Evaluate proactive social triggers from normalized sensor observations.

This module defines the social-trigger domain model, bounded threshold config,
and the stateful engine that turns recent vision, audio, and PIR signals into
one candidate proactive prompt.
"""

from __future__ import annotations

import math  # AUDIT-FIX(#2,#4): Finite/monotonic timestamp checks and numeric threshold sanitisation require explicit numeric validation.
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from twinr.agent.base_agent.config import TwinrConfig

from .scoring import TriggerScoreEvidence, bool_score, hold_progress, recent_progress, weighted_trigger_score


# AUDIT-FIX(#2,#4,#5): Normalize permissive upstream sensor/config payloads before they can corrupt state, crash enum access, or break scoring math.
def _coerce_bool(value: object, *, default: bool = False) -> bool:
    """Coerce one value to a boolean with fallback."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_optional_bool(value: object) -> bool | None:
    """Coerce one value to ``bool`` or ``None``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_timestamp(value: object) -> float | None:
    """Coerce one value to a non-negative finite timestamp."""

    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0.0:
        return None
    return timestamp


def _normalize_positive_float(value: object, *, default: float) -> float:
    """Coerce one value to a positive finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


def _normalize_non_negative_float(value: object, *, default: float = 0.0) -> float:
    """Coerce one value to a non-negative finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0.0:
        return default
    return number


def _coerce_recent_age(value: object) -> float | None:
    """Coerce one age-like value to a non-negative finite float."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _coerce_optional_text(value: object, *, limit: int) -> str | None:
    """Coerce one optional text field into bounded ASCII-safe text."""

    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    return text[:limit]


def _coerce_optional_azimuth(value: object) -> int | None:
    """Coerce one azimuth value into ``0..359`` or ``None``."""

    if value is None:
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number % 360


def _normalize_unit_interval(value: object, *, default: float) -> float:
    """Clamp one numeric value into ``[0.0, 1.0]`` with fallback."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


class SocialBodyPose(StrEnum):
    """Describe the coarse body-pose classes used by trigger logic."""

    UNKNOWN = "unknown"
    UPRIGHT = "upright"
    SEATED = "seated"
    SLUMPED = "slumped"
    LYING_LOW = "lying_low"
    FLOOR = "floor"


class SocialPersonZone(StrEnum):
    """Describe the coarse horizontal zone of the primary visible person."""

    UNKNOWN = "unknown"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class SocialGestureEvent(StrEnum):
    """Describe the tiny gesture vocabulary exposed to policy consumers."""

    NONE = "none"
    STOP = "stop"
    DISMISS = "dismiss"
    CONFIRM = "confirm"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SocialSpatialBox:
    """Describe one normalized bounding box in ``top,left,bottom,right`` order."""

    top: float
    left: float
    bottom: float
    right: float

    def __post_init__(self) -> None:
        """Clamp coordinates into bounds and enforce monotonic edges."""

        top = _normalize_unit_interval(self.top, default=0.0)
        left = _normalize_unit_interval(self.left, default=0.0)
        bottom = _normalize_unit_interval(self.bottom, default=top)
        right = _normalize_unit_interval(self.right, default=left)
        if bottom < top:
            bottom = top
        if right < left:
            right = left
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "bottom", bottom)
        object.__setattr__(self, "right", right)

    @property
    def center_x(self) -> float:
        """Return the normalized horizontal center."""

        return (self.left + self.right) / 2.0

    @property
    def center_y(self) -> float:
        """Return the normalized vertical center."""

        return (self.top + self.bottom) / 2.0

    @property
    def area(self) -> float:
        """Return the normalized area."""

        return max(0.0, self.bottom - self.top) * max(0.0, self.right - self.left)


@dataclass(frozen=True, slots=True)
class SocialDetectedObject:
    """Describe one explicit object detection for downstream policy consumers."""

    label: str
    confidence: float
    zone: SocialPersonZone = SocialPersonZone.UNKNOWN
    stable: bool = False
    box: SocialSpatialBox | None = None

    def __post_init__(self) -> None:
        """Normalize object metadata into bounded, inspectable values."""

        label = str(self.label or "").strip().lower().replace(" ", "_")
        if not label:
            label = "unknown"
        object.__setattr__(self, "label", label[:64])
        object.__setattr__(self, "confidence", _normalize_unit_interval(self.confidence, default=0.0))
        object.__setattr__(self, "zone", _coerce_person_zone(self.zone))
        object.__setattr__(self, "stable", _coerce_bool(self.stable))
        object.__setattr__(self, "box", _coerce_spatial_box(self.box))


def _coerce_body_pose(value: object) -> SocialBodyPose:
    """Coerce one value to a known body pose."""

    if isinstance(value, SocialBodyPose):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return SocialBodyPose(normalized)
        except ValueError:
            return SocialBodyPose.UNKNOWN
    return SocialBodyPose.UNKNOWN


def _coerce_gesture_event(value: object) -> SocialGestureEvent:
    """Coerce one value to a known gesture event."""

    if isinstance(value, SocialGestureEvent):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return SocialGestureEvent(normalized)
        except ValueError:
            return SocialGestureEvent.UNKNOWN
    return SocialGestureEvent.UNKNOWN


def _coerce_person_zone(value: object) -> SocialPersonZone:
    """Coerce one value to a known coarse person zone."""

    if isinstance(value, SocialPersonZone):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return SocialPersonZone(normalized)
        except ValueError:
            return SocialPersonZone.UNKNOWN
    return SocialPersonZone.UNKNOWN


def _coerce_non_negative_int(value: object, *, default: int) -> int:
    """Coerce one value to a non-negative integer with fallback."""

    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def _coerce_bounded_text(value: object, *, max_length: int = 160) -> str | None:
    """Coerce one value to bounded operator-safe text."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_length]


def _coerce_spatial_box(value: object) -> SocialSpatialBox | None:
    """Coerce one value to one normalized spatial box."""

    if isinstance(value, SocialSpatialBox):
        return value
    if isinstance(value, dict):
        candidate = (
            value.get("top"),
            value.get("left"),
            value.get("bottom"),
            value.get("right"),
        )
    elif isinstance(value, (tuple, list)) and len(value) == 4:
        candidate = tuple(value)
    else:
        candidate = None
    if candidate is None:
        return None
    try:
        top, left, bottom, right = (float(item) for item in candidate)
    except (TypeError, ValueError):
        return None
    return SocialSpatialBox(top=top, left=left, bottom=bottom, right=right)


def _coerce_optional_ratio(value: object) -> float | None:
    """Coerce one optional ratio value into ``[0.0, 1.0]``."""

    if value is None:
        return None
    return _normalize_unit_interval(value, default=0.0)


def _coerce_detected_object(value: object) -> SocialDetectedObject | None:
    """Coerce one object-like payload to ``SocialDetectedObject``."""

    if isinstance(value, SocialDetectedObject):
        return value
    if isinstance(value, dict):
        return SocialDetectedObject(
            label=value.get("label", ""),
            confidence=value.get("confidence", 0.0),
            zone=value.get("zone", SocialPersonZone.UNKNOWN),
            stable=value.get("stable", False),
            box=value.get("box"),
        )
    return None


def _coerce_detected_objects(value: object) -> tuple[SocialDetectedObject, ...]:
    """Coerce one iterable payload to a tuple of detected objects."""

    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(item, SocialDetectedObject) for item in value):
        return value
    if not isinstance(value, (tuple, list)):
        return ()
    items: list[SocialDetectedObject] = []
    for item in value:
        detected = _coerce_detected_object(item)
        if detected is not None:
            items.append(detected)
    return tuple(items)


def _is_floor_like_pose(body_pose: SocialBodyPose) -> bool:
    """Return whether one coarse pose should count as floor-like."""

    return body_pose in {SocialBodyPose.LYING_LOW, SocialBodyPose.FLOOR}


def _is_upright_like_pose(body_pose: SocialBodyPose) -> bool:
    """Return whether one pose should count as upright for transitions."""

    return body_pose in {SocialBodyPose.UPRIGHT, SocialBodyPose.SEATED, SocialBodyPose.SLUMPED}


class SocialTriggerPriority(IntEnum):
    """Order social triggers from least to most urgent."""

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
    """Describe one normalized vision observation tick."""

    person_visible: bool = False
    person_count: int = 0
    person_recently_visible: bool | None = None
    person_appeared_at: float | None = None
    person_disappeared_at: float | None = None
    primary_person_zone: SocialPersonZone = SocialPersonZone.UNKNOWN
    primary_person_box: SocialSpatialBox | None = None
    primary_person_center_x: float | None = None
    primary_person_center_y: float | None = None
    looking_toward_device: bool = False
    person_near_device: bool | None = None
    engaged_with_device: bool | None = None
    visual_attention_score: float | None = None
    body_pose: SocialBodyPose = SocialBodyPose.UNKNOWN
    pose_confidence: float | None = None
    body_state_changed_at: float | None = None
    smiling: bool = False
    hand_or_object_near_camera: bool = False
    showing_intent_likely: bool | None = None
    showing_intent_started_at: float | None = None
    gesture_event: SocialGestureEvent = SocialGestureEvent.NONE
    gesture_confidence: float | None = None
    objects: tuple[SocialDetectedObject, ...] = ()
    camera_online: bool = False
    camera_ready: bool = False
    camera_ai_ready: bool = False
    camera_error: str | None = None
    last_camera_frame_at: float | None = None
    last_camera_health_change_at: float | None = None


@dataclass(frozen=True, slots=True)
class SocialAudioObservation:
    """Describe one normalized audio observation tick."""

    speech_detected: bool | None = None
    distress_detected: bool | None = None
    room_quiet: bool | None = None
    recent_speech_age_s: float | None = None
    azimuth_deg: int | None = None
    device_runtime_mode: str | None = None
    signal_source: str | None = None
    host_control_ready: bool | None = None
    transport_reason: str | None = None
    mute_active: bool | None = None


@dataclass(frozen=True, slots=True)
class SocialObservation:
    """Combine normalized sensor observations for one trigger-engine tick."""

    observed_at: float
    inspected: bool = False  # AUDIT-FIX(#1): Absent vision data must default to "unknown", not to an authoritative "no person visible".
    pir_motion_detected: bool = False
    low_motion: bool = False
    vision: SocialVisionObservation = field(default_factory=SocialVisionObservation)
    audio: SocialAudioObservation = field(default_factory=SocialAudioObservation)


@dataclass(frozen=True, slots=True)
class SocialTriggerDecision:
    """Describe one emitted proactive social trigger."""

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
    """Describe one scored trigger candidate before selection."""

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
    """Store hold windows and score thresholds for social triggers."""

    person_returned_absence_s: float = 20.0 * 60.0
    person_returned_recent_motion_s: float = 30.0
    attention_window_s: float = 6.0
    slumped_quiet_s: float = 20.0
    possible_fall_stillness_s: float = 10.0
    possible_fall_visibility_loss_hold_s: float = 15.0
    possible_fall_visibility_loss_arming_s: float = 6.0
    possible_fall_slumped_visibility_loss_arming_s: float = 4.0
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

    def __post_init__(self) -> None:
        """Normalize duration and score thresholds after construction."""

        # AUDIT-FIX(#4): Reject non-finite/zero/negative timing inputs and clamp score thresholds to a valid unit interval before any scoring math runs.
        duration_defaults = {
            "person_returned_absence_s": 20.0 * 60.0,
            "person_returned_recent_motion_s": 30.0,
            "attention_window_s": 6.0,
            "slumped_quiet_s": 20.0,
            "possible_fall_stillness_s": 10.0,
            "possible_fall_visibility_loss_hold_s": 15.0,
            "possible_fall_visibility_loss_arming_s": 6.0,
            "possible_fall_slumped_visibility_loss_arming_s": 4.0,
            "floor_stillness_s": 20.0,
            "showing_intent_hold_s": 1.5,
            "positive_contact_hold_s": 1.5,
            "distress_hold_s": 3.0,
            "fall_transition_window_s": 8.0,
        }
        score_defaults = {
            "person_returned_score_threshold": 0.9,
            "attention_window_score_threshold": 0.86,
            "slumped_quiet_score_threshold": 0.9,
            "possible_fall_score_threshold": 0.82,
            "floor_stillness_score_threshold": 0.9,
            "showing_intent_score_threshold": 0.84,
            "positive_contact_score_threshold": 0.84,
            "distress_possible_score_threshold": 0.85,
        }
        for name, default in duration_defaults.items():
            object.__setattr__(self, name, _normalize_positive_float(getattr(self, name), default=default))
        for name, default in score_defaults.items():
            object.__setattr__(self, name, _normalize_unit_interval(getattr(self, name), default=default))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerThresholds":
        """Build thresholds from ``TwinrConfig`` with safe defaults."""

        defaults = cls()
        # AUDIT-FIX(#3,#4): Fall back to baked-in defaults when older TwinrConfig objects or partial .env payloads lack new fields or provide malformed values.
        return cls(
            person_returned_absence_s=getattr(config, "proactive_person_returned_absence_s", defaults.person_returned_absence_s),
            person_returned_recent_motion_s=getattr(config, "proactive_person_returned_recent_motion_s", defaults.person_returned_recent_motion_s),
            attention_window_s=getattr(config, "proactive_attention_window_s", defaults.attention_window_s),
            slumped_quiet_s=getattr(config, "proactive_slumped_quiet_s", defaults.slumped_quiet_s),
            possible_fall_stillness_s=getattr(config, "proactive_possible_fall_stillness_s", defaults.possible_fall_stillness_s),
            possible_fall_visibility_loss_hold_s=getattr(
                config,
                "proactive_possible_fall_visibility_loss_hold_s",
                defaults.possible_fall_visibility_loss_hold_s,
            ),
            possible_fall_visibility_loss_arming_s=getattr(
                config,
                "proactive_possible_fall_visibility_loss_arming_s",
                defaults.possible_fall_visibility_loss_arming_s,
            ),
            possible_fall_slumped_visibility_loss_arming_s=getattr(
                config,
                "proactive_possible_fall_slumped_visibility_loss_arming_s",
                defaults.possible_fall_slumped_visibility_loss_arming_s,
            ),
            floor_stillness_s=getattr(config, "proactive_floor_stillness_s", defaults.floor_stillness_s),
            showing_intent_hold_s=getattr(config, "proactive_showing_intent_hold_s", defaults.showing_intent_hold_s),
            positive_contact_hold_s=getattr(config, "proactive_positive_contact_hold_s", defaults.positive_contact_hold_s),
            distress_hold_s=getattr(config, "proactive_distress_hold_s", defaults.distress_hold_s),
            fall_transition_window_s=getattr(config, "proactive_fall_transition_window_s", defaults.fall_transition_window_s),
            person_returned_score_threshold=getattr(
                config,
                "proactive_person_returned_score_threshold",
                defaults.person_returned_score_threshold,
            ),
            attention_window_score_threshold=getattr(
                config,
                "proactive_attention_window_score_threshold",
                defaults.attention_window_score_threshold,
            ),
            slumped_quiet_score_threshold=getattr(
                config,
                "proactive_slumped_quiet_score_threshold",
                defaults.slumped_quiet_score_threshold,
            ),
            possible_fall_score_threshold=getattr(
                config,
                "proactive_possible_fall_score_threshold",
                defaults.possible_fall_score_threshold,
            ),
            floor_stillness_score_threshold=getattr(
                config,
                "proactive_floor_stillness_score_threshold",
                defaults.floor_stillness_score_threshold,
            ),
            showing_intent_score_threshold=getattr(
                config,
                "proactive_showing_intent_score_threshold",
                defaults.showing_intent_score_threshold,
            ),
            positive_contact_score_threshold=getattr(
                config,
                "proactive_positive_contact_score_threshold",
                defaults.positive_contact_score_threshold,
            ),
            distress_possible_score_threshold=getattr(
                config,
                "proactive_distress_possible_score_threshold",
                defaults.distress_possible_score_threshold,
            ),
        )


class SocialTriggerEngine:
    """Track recent social signals and emit the strongest passing trigger.

    The engine maintains bounded state across observation ticks so it can
    reason about visibility gaps, holds, cooldowns, and fall-like transitions
    without mixing hardware acquisition into the scoring logic.
    """

    def __init__(
        self,
        *,
        user_name: str | None = None,
        thresholds: SocialTriggerThresholds | None = None,
    ) -> None:
        """Initialize one stateful trigger engine."""

        # AUDIT-FIX(#5): Normalise potentially non-string display names coming from config instead of assuming .strip() exists.
        self.user_name = None if user_name is None else (str(user_name).strip() or None)
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
        self._consecutive_visible_inspected_count: int = 0
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
        self._last_observed_at: float | None = None  # AUDIT-FIX(#2): Guard against backwards/non-finite clocks corrupting duration state.

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "SocialTriggerEngine":
        """Build one trigger engine from the canonical Twinr config."""

        return cls(
            user_name=getattr(config, "user_display_name", None),  # AUDIT-FIX(#3,#5): Preserve startup compatibility with older config objects.
            thresholds=SocialTriggerThresholds.from_config(config),
        )

    @property
    def last_evaluations(self) -> tuple[SocialTriggerEvaluation, ...]:
        """Return the scored candidates from the latest observation tick."""

        return self._last_evaluations

    @property
    def best_evaluation(self) -> SocialTriggerEvaluation | None:
        """Return the strongest latest candidate, including near misses."""

        if not self._last_evaluations:
            return None
        passed_candidates = tuple(item for item in self._last_evaluations if item.passed)
        if passed_candidates:
            return max(
                passed_candidates,
                key=lambda item: (int(item.priority), item.score),  # AUDIT-FIX(#7): Keep diagnostics aligned with actual trigger selection order.
            )
        return max(
            self._last_evaluations,
            key=lambda item: (item.score, int(item.priority)),  # AUDIT-FIX(#9): Surface the strongest near miss when no trigger passed.
        )

    def observe(self, observation: SocialObservation) -> SocialTriggerDecision | None:
        """Update state from one observation tick and emit a trigger if ready.

        Args:
            observation: One normalized sensor observation for the current tick.

        Returns:
            The selected proactive trigger decision, or ``None`` when no
            candidate passes validation, scoring, and cooldown checks.
        """

        # AUDIT-FIX(#2,#5): Reject malformed or stale observations before mutating internal state; this prevents crashes and time-warped cooldown logic.
        if not isinstance(observation, SocialObservation):
            self._last_evaluations = ()
            return None
        now = _coerce_timestamp(observation.observed_at)
        if now is None:
            self._last_evaluations = ()
            return None
        if self._last_observed_at is not None and now < self._last_observed_at:
            self._last_evaluations = ()
            return None
        self._last_observed_at = now

        inspected = _coerce_bool(observation.inspected)
        pir_motion_detected = _coerce_bool(observation.pir_motion_detected)
        low_motion = _coerce_bool(observation.low_motion)
        raw_vision = observation.vision
        vision = self._normalize_vision(observation.vision, inspected=inspected)
        audio = self._normalize_audio(observation.audio)
        sensor_absence_signal = self._should_treat_as_sensor_absence_signal(
            inspected=inspected,
            pir_motion_detected=pir_motion_detected,
            vision=raw_vision,
        )

        if pir_motion_detected:
            self._last_pir_motion_at = now

        absence_duration = self._update_presence_state(
            now,
            inspected=inspected,
            person_visible=vision.person_visible,
            sensor_absence_signal=sensor_absence_signal,
        )
        self._update_confirmed_visibility_state(
            inspected=inspected,
            person_visible=vision.person_visible,
        )
        self._update_pose_state(
            now,
            inspected=inspected,
            person_visible=vision.person_visible,
            body_pose=vision.body_pose,
        )
        self._quiet_since = self._next_since(audio.speech_detected is False, self._quiet_since, now)
        self._looking_since = self._next_since(
            inspected and vision.person_visible and vision.looking_toward_device,
            self._looking_since,
            now,
        )
        self._slumped_since = self._next_since(
            inspected and vision.person_visible and vision.body_pose == SocialBodyPose.SLUMPED,
            self._slumped_since,
            now,
        )
        self._floor_since = self._next_since(
            inspected and vision.person_visible and _is_floor_like_pose(vision.body_pose),
            self._floor_since,
            now,
        )
        self._low_motion_since = self._next_since(low_motion, self._low_motion_since, now)
        self._showing_since = self._next_since(
            inspected
            and vision.person_visible
            and (
                vision.showing_intent_likely is True
                or (vision.looking_toward_device and vision.hand_or_object_near_camera)
            ),
            self._showing_since,
            now,
        )
        self._smile_since = self._next_since(
            inspected and vision.person_visible and vision.looking_toward_device and vision.smiling,
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
        """Score whether a person recently returned after tracked absence."""

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
        """Score whether the device has a short quiet attention window."""

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
        """Score whether a visible person remains slumped, quiet, and still."""

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
        """Score whether recent state suggests a possible fall."""

        low_motion_since = self._possible_fall_post_transition_since(now, self._low_motion_since)
        quiet_since = self._possible_fall_post_transition_since(now, self._quiet_since)
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
                value=hold_progress(now, low_motion_since, self.thresholds.possible_fall_stillness_s),
                weight=0.28,
                detail=self._hold_detail(low_motion_since, now, self.thresholds.possible_fall_stillness_s),
            ),
            TriggerScoreEvidence(
                key="quiet_hold",
                value=hold_progress(now, quiet_since, self.thresholds.possible_fall_stillness_s),
                weight=0.15,
                detail=self._hold_detail(quiet_since, now, self.thresholds.possible_fall_stillness_s),
            ),
        )
        blocked_reason = None
        if self._cooldown_active("possible_fall", now):
            blocked_reason = "cooldown_active"
        elif self._possible_fall_loss_candidate_at is not None and not self._possible_fall_visibility_loss_hold_complete(now):
            blocked_reason = "visibility_loss_hold_incomplete"
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
        """Score whether a person remains low to the floor and still."""

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
        """Score whether the person seems to be showing something to the device."""

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
        """Score whether distress-like audio aligns with concerning posture."""

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
        """Score whether the person is positively engaged with the device."""

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

    def _update_presence_state(
        self,
        now: float,
        *,
        inspected: bool,
        person_visible: bool,
        sensor_absence_signal: bool = False,
    ) -> float | None:
        """Update tracked presence and return any completed absence duration."""

        # AUDIT-FIX(#1,#10): Uninspected ticks normally freeze presence state, except for the coordinator's explicit sensor-only absence path.
        if not inspected and not sensor_absence_signal:
            return None

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

    def _should_treat_as_sensor_absence_signal(
        self,
        *,
        inspected: bool,
        pir_motion_detected: bool,
        vision: object,
    ) -> bool:
        """Return whether sensor-only input should extend the absence state."""

        if inspected or pir_motion_detected:
            return False
        if not isinstance(vision, SocialVisionObservation):
            return False
        if _coerce_bool(vision.person_visible):
            return False
        return self._person_visible or self._absence_started_at is not None

    def _update_pose_state(
        self,
        now: float,
        *,
        inspected: bool,
        person_visible: bool,
        body_pose: SocialBodyPose,
    ) -> None:
        """Update tracked pose and fall-transition state."""

        # AUDIT-FIX(#1): Only inspected, person-present frames may mutate pose state; otherwise stale/unknown vision drops can fabricate transitions.
        if not inspected or not person_visible:
            return
        if _is_upright_like_pose(body_pose):
            self._last_non_floor_pose_at = now
        if body_pose == SocialBodyPose.SLUMPED:
            self._last_slumped_at = now
        if body_pose == self._current_pose:
            return
        self._current_pose = body_pose
        if _is_floor_like_pose(body_pose):
            if (
                self._last_non_floor_pose_at is not None
                and (now - self._last_non_floor_pose_at) <= self.thresholds.fall_transition_window_s
            ):
                self._possible_fall_candidate_at = now
        else:
            self._possible_fall_candidate_at = None

    def _cooldown_active(self, trigger_id: str, now: float) -> bool:
        """Return whether one trigger is still on cooldown."""

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
        """Build one scored trigger evaluation from evidence."""

        # AUDIT-FIX(#4,#6): Clamp thresholds and renormalise evidence so bad config or overweighted evidence cannot skew or explode scoring.
        score_card = weighted_trigger_score(
            threshold=_normalize_unit_interval(threshold, default=1.0),
            evidence=self._normalize_evidence(evidence),
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
        """Format one prompt variant with the configured display name."""

        if self.user_name is None:
            return base
        return with_name.format(name=self.user_name)

    def _conjunction_since(self, *starts: float | None) -> float | None:
        """Return the shared activation start when all inputs are active."""

        active_starts = [start for start in starts if start is not None]
        if len(active_starts) != len(starts):
            return None
        return max(active_starts)

    def _next_since(self, active: bool, current: float | None, now: float) -> float | None:
        """Advance one hold start timestamp when its signal remains active."""

        if not active:
            return None
        if current is None:
            return now
        return current

    def _hold_detail(self, since: float | None, now: float, target_s: float) -> str:
        """Render one human-readable hold-progress detail string."""

        if since is None:
            return f"active_for=0.0s target={target_s:.1f}s"
        return f"active_for={max(0.0, now - since):.1f}s target={target_s:.1f}s"  # AUDIT-FIX(#2): Never emit negative durations after rejected/stale time inputs.

    def _fell_out_of_view_after_fall_like_presence(self, now: float, *, visible_duration: float | None) -> bool:
        """Return whether recent visibility loss looks fall-like."""

        if self._possible_fall_candidate_at is not None:
            return True
        if self._consecutive_visible_inspected_count < 2:
            return False
        if visible_duration is None or visible_duration < self._fall_visibility_loss_arming_s():
            return False
        if self._current_pose == SocialBodyPose.SLUMPED and self._last_slumped_at is not None:
            return (
                (now - self._last_slumped_at) <= self.thresholds.fall_transition_window_s
                and self._slumped_hold_before_visibility_loss(now) >= self._fall_slumped_visibility_loss_arming_s()
            )
        if self._current_pose == SocialBodyPose.UNKNOWN and self._last_slumped_at is not None:
            return (
                (now - self._last_slumped_at) <= self.thresholds.fall_transition_window_s
                and self._slumped_hold_before_visibility_loss(now) >= self._fall_slumped_visibility_loss_arming_s()
            )
        return False

    def _possible_fall_transition_anchor(self) -> float | None:
        """Return the newest recorded fall-transition anchor."""

        anchors = [
            anchor
            for anchor in (
                self._possible_fall_candidate_at,
                self._possible_fall_loss_candidate_at,
            )
            if anchor is not None
        ]
        if not anchors:
            return None
        return max(anchors)

    def _possible_fall_post_transition_since(
        self,
        now: float,
        signal_since: float | None,
    ) -> float | None:
        """Clamp one hold start to the relevant fall-transition anchor."""

        transition_anchor = self._possible_fall_transition_anchor()
        if transition_anchor is None or signal_since is None:
            return None
        if signal_since > now:
            return now
        return max(signal_since, transition_anchor)

    def _possible_fall_transition_signal(self) -> float:
        """Return the weighted signal strength of the current fall transition."""

        if self._possible_fall_candidate_at is not None:
            return 1.0
        if self._possible_fall_loss_candidate_at is None:
            return 0.0
        if self._possible_fall_loss_pose == SocialBodyPose.SLUMPED:
            return 0.9
        if self._possible_fall_loss_pose in {SocialBodyPose.UPRIGHT, SocialBodyPose.SEATED}:
            return 0.75
        return 0.55

    def _possible_fall_low_or_missing_hold(self, now: float) -> float:
        """Return the stronger of floor hold and visibility-loss hold."""

        floor_hold = hold_progress(now, self._floor_since, self.thresholds.possible_fall_stillness_s)
        missing_hold = hold_progress(
            now,
            self._possible_fall_loss_candidate_at,
            self.thresholds.possible_fall_visibility_loss_hold_s,
        )
        return max(floor_hold, missing_hold)

    def _possible_fall_low_or_missing_detail(self, now: float) -> str:
        """Render floor and visibility-loss hold details."""

        floor_detail = self._hold_detail(self._floor_since, now, self.thresholds.possible_fall_stillness_s)
        missing_detail = self._hold_detail(
            self._possible_fall_loss_candidate_at,
            now,
            self.thresholds.possible_fall_visibility_loss_hold_s,
        )
        return f"floor={floor_detail}; missing={missing_detail}"

    def _possible_fall_visibility_loss_hold_complete(self, now: float) -> bool:
        """Return whether the visibility-loss hold has fully armed."""

        if self._possible_fall_loss_candidate_at is None:
            return True
        return (
            hold_progress(
                now,
                self._possible_fall_loss_candidate_at,
                self.thresholds.possible_fall_visibility_loss_hold_s,
            )
            >= 1.0
        )

    def _possible_fall_transition_detail(self, now: float) -> str:
        """Render one detail string for the active fall-transition signal."""

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
        """Return the visibility-loss arming threshold in seconds."""

        return max(0.0, self.thresholds.possible_fall_visibility_loss_arming_s)

    def _fall_slumped_visibility_loss_arming_s(self) -> float:
        """Return the slumped-loss arming threshold in seconds."""

        return max(0.0, self.thresholds.possible_fall_slumped_visibility_loss_arming_s)

    def _slumped_hold_before_visibility_loss(self, now: float) -> float:
        """Return how long the person was slumped before visibility loss."""

        if self._slumped_since is None:
            return 0.0
        return max(0.0, now - self._slumped_since)

    def _update_confirmed_visibility_state(self, *, inspected: bool, person_visible: bool) -> None:
        """Update the bounded consecutive-visible counter."""

        if not inspected:
            return
        if person_visible:
            self._consecutive_visible_inspected_count = min(
                2,
                self._consecutive_visible_inspected_count + 1,
            )  # AUDIT-FIX(#1): Only "0, 1, 2+" matters; cap the counter instead of growing forever during long uptimes.
            return
        self._consecutive_visible_inspected_count = 0

    def _seconds(self, value: float | None) -> str:
        """Format one optional duration in seconds."""

        if value is None:
            return "n/a"
        return f"{max(0.0, value):.1f}s"  # AUDIT-FIX(#2): Defensive formatting if upstream clocks jitter backward and the observation gets dropped.

    def _normalize_audio(self, audio: object) -> SocialAudioObservation:
        """Normalize one audio payload to the strict internal type."""

        # AUDIT-FIX(#5): Coerce permissive payloads into the strict internal type expected by trigger logic.
        if not isinstance(audio, SocialAudioObservation):
            return SocialAudioObservation()
        return SocialAudioObservation(
            speech_detected=_coerce_optional_bool(audio.speech_detected),
            distress_detected=_coerce_optional_bool(audio.distress_detected),
            room_quiet=_coerce_optional_bool(getattr(audio, "room_quiet", None)),
            recent_speech_age_s=_coerce_recent_age(getattr(audio, "recent_speech_age_s", None)),
            azimuth_deg=_coerce_optional_azimuth(getattr(audio, "azimuth_deg", None)),
            device_runtime_mode=_coerce_optional_text(getattr(audio, "device_runtime_mode", None), limit=64),
            signal_source=_coerce_optional_text(getattr(audio, "signal_source", None), limit=64),
            host_control_ready=_coerce_optional_bool(getattr(audio, "host_control_ready", None)),
            transport_reason=_coerce_optional_text(getattr(audio, "transport_reason", None), limit=120),
            mute_active=_coerce_optional_bool(getattr(audio, "mute_active", None)),
        )

    def _normalize_vision(
        self,
        vision: object,
        *,
        inspected: bool,
    ) -> SocialVisionObservation:
        """Normalize one vision payload to the strict internal type."""

        # AUDIT-FIX(#1,#5): Treat "not inspected" as unknown vision and normalise malformed pose/bool payloads before they reach safety logic.
        if not inspected or not isinstance(vision, SocialVisionObservation):
            return SocialVisionObservation()

        person_visible = _coerce_bool(vision.person_visible)
        primary_person_box = (
            _coerce_spatial_box(getattr(vision, "primary_person_box", None))
            if person_visible
            else None
        )
        primary_person_center_x = (
            primary_person_box.center_x
            if primary_person_box is not None
            else _coerce_optional_ratio(getattr(vision, "primary_person_center_x", None))
        )
        primary_person_center_y = (
            primary_person_box.center_y
            if primary_person_box is not None
            else _coerce_optional_ratio(getattr(vision, "primary_person_center_y", None))
        )
        return SocialVisionObservation(
            person_visible=person_visible,
            person_count=(
                max(1, _coerce_non_negative_int(getattr(vision, "person_count", 1), default=1))
                if person_visible
                else 0
            ),
            person_recently_visible=_coerce_optional_bool(getattr(vision, "person_recently_visible", None)),
            person_appeared_at=_coerce_timestamp(getattr(vision, "person_appeared_at", None)),
            person_disappeared_at=_coerce_timestamp(getattr(vision, "person_disappeared_at", None)),
            primary_person_zone=(
                _coerce_person_zone(getattr(vision, "primary_person_zone", SocialPersonZone.UNKNOWN))
                if person_visible
                else SocialPersonZone.UNKNOWN
            ),
            primary_person_box=primary_person_box,
            primary_person_center_x=(primary_person_center_x if person_visible else None),
            primary_person_center_y=(primary_person_center_y if person_visible else None),
            looking_toward_device=person_visible and _coerce_bool(vision.looking_toward_device),
            person_near_device=_coerce_optional_bool(getattr(vision, "person_near_device", None)),
            engaged_with_device=_coerce_optional_bool(getattr(vision, "engaged_with_device", None)),
            visual_attention_score=_coerce_optional_ratio(getattr(vision, "visual_attention_score", None)),
            body_pose=(_coerce_body_pose(vision.body_pose) if person_visible else SocialBodyPose.UNKNOWN),
            pose_confidence=_coerce_optional_ratio(getattr(vision, "pose_confidence", None)),
            body_state_changed_at=_coerce_timestamp(getattr(vision, "body_state_changed_at", None)),
            smiling=person_visible and _coerce_bool(vision.smiling),
            hand_or_object_near_camera=_coerce_bool(vision.hand_or_object_near_camera),
            showing_intent_likely=_coerce_optional_bool(getattr(vision, "showing_intent_likely", None)),
            showing_intent_started_at=_coerce_timestamp(getattr(vision, "showing_intent_started_at", None)),
            gesture_event=_coerce_gesture_event(getattr(vision, "gesture_event", SocialGestureEvent.NONE)),
            gesture_confidence=_coerce_optional_ratio(getattr(vision, "gesture_confidence", None)),
            objects=_coerce_detected_objects(getattr(vision, "objects", ())),
            camera_online=_coerce_bool(getattr(vision, "camera_online", False)),
            camera_ready=_coerce_bool(getattr(vision, "camera_ready", False)),
            camera_ai_ready=_coerce_bool(getattr(vision, "camera_ai_ready", False)),
            camera_error=_coerce_bounded_text(getattr(vision, "camera_error", None)),
            last_camera_frame_at=_coerce_timestamp(getattr(vision, "last_camera_frame_at", None)),
            last_camera_health_change_at=_coerce_timestamp(
                getattr(vision, "last_camera_health_change_at", None)
            ),
        )

    def _normalize_evidence(
        self,
        evidence: tuple[TriggerScoreEvidence, ...],
    ) -> tuple[TriggerScoreEvidence, ...]:
        """Normalize evidence weights into one stable tuple."""

        sanitized: list[tuple[str, float, float, str]] = []
        total_weight = 0.0
        for item in evidence:
            key = str(getattr(item, "key", ""))
            value = _normalize_unit_interval(getattr(item, "value", 0.0), default=0.0)
            weight = _normalize_non_negative_float(getattr(item, "weight", 0.0), default=0.0)
            detail = str(getattr(item, "detail", ""))
            sanitized.append((key, value, weight, detail))
            total_weight += weight

        if not sanitized:
            return ()

        normalized_weight = 1.0 / len(sanitized) if total_weight <= 0.0 else None
        return tuple(
            TriggerScoreEvidence(
                key=key,
                value=value,
                weight=(normalized_weight if normalized_weight is not None else weight / total_weight),
                detail=detail,
            )
            for key, value, weight, detail in sanitized
        )


__all__ = [
    "SocialAudioObservation",
    "SocialBodyPose",
    "SocialDetectedObject",
    "SocialGestureEvent",
    "SocialObservation",
    "SocialPersonZone",
    "SocialSpatialBox",
    "SocialTriggerDecision",
    "SocialTriggerEngine",
    "SocialTriggerEvaluation",
    "SocialTriggerPriority",
    "SocialTriggerThresholds",
    "SocialVisionObservation",
]
