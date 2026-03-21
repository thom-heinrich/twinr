"""Stabilize camera observations into automation-friendly facts and events.

This module turns raw ``SocialVisionObservation`` ticks into a bounded camera
snapshot and a small rising-edge event surface. It owns debounce, cooldown,
object stability, and unknown-state handling for camera-derived automation
signals so runtime orchestrators can stay thin.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .engine import (
    SocialBodyPose,
    SocialDetectedObject,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisionObservation,
)

_PERSON_VISIBLE_EVENT = "camera.person_visible"
_HAND_NEAR_EVENT = "camera.hand_or_object_near_camera"
_PERSON_RETURNED_EVENT = "camera.person_returned"
_ATTENTION_WINDOW_EVENT = "camera.attention_window_opened"
_SHOWING_INTENT_EVENT = "camera.showing_intent_started"
_MOTION_EVENT = "camera.motion_changed"
_GESTURE_EVENT = "camera.gesture_detected"
_COARSE_ARM_GESTURE_EVENT = "camera.coarse_arm_gesture_detected"
_FINE_HAND_GESTURE_EVENT = "camera.fine_hand_gesture_detected"
_OBJECT_STABLE_EVENT = "camera.object_detected_stable"
_EXPLICIT_FINE_HAND_GESTURES = frozenset(
    {
        SocialFineHandGesture.THUMBS_UP,
        SocialFineHandGesture.THUMBS_DOWN,
        SocialFineHandGesture.POINTING,
        SocialFineHandGesture.OK_SIGN,
    }
)
_MIN_CENTER_SMOOTHING_ALPHA = 0.1
_MAX_CENTER_SMOOTHING_ALPHA = 1.0
_MIN_CENTER_DEADBAND = 0.0
_MAX_CENTER_DEADBAND = 0.25
_MIN_CENTER_SMOOTHING_WINDOW_S = 0.1
_MAX_CENTER_SMOOTHING_WINDOW_S = 10.0


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceConfig:
    """Store stabilization rules for camera-derived automation signals."""

    person_visible_on_samples: int = 1
    person_visible_off_samples: int = 2
    person_visible_unknown_hold_s: float = 9.0
    person_visible_event_cooldown_s: float = 9.0
    person_recently_visible_window_s: float = 30.0
    person_returned_absence_s: float = 20.0 * 60.0
    looking_toward_device_on_samples: int = 1
    looking_toward_device_off_samples: int = 2
    looking_toward_device_unknown_hold_s: float = 9.0
    person_near_device_on_samples: int = 1
    person_near_device_off_samples: int = 2
    person_near_device_unknown_hold_s: float = 9.0
    engaged_with_device_on_samples: int = 1
    engaged_with_device_off_samples: int = 2
    engaged_with_device_unknown_hold_s: float = 9.0
    showing_intent_on_samples: int = 1
    showing_intent_off_samples: int = 2
    showing_intent_unknown_hold_s: float = 9.0
    showing_intent_event_cooldown_s: float = 9.0
    hand_or_object_near_camera_on_samples: int = 1
    hand_or_object_near_camera_off_samples: int = 2
    hand_or_object_near_camera_unknown_hold_s: float = 9.0
    hand_or_object_near_camera_event_cooldown_s: float = 9.0
    motion_event_cooldown_s: float = 9.0
    gesture_event_cooldown_s: float = 9.0
    fine_hand_explicit_hold_s: float = 0.45
    primary_person_center_smoothing_alpha: float = 0.58
    primary_person_center_deadband: float = 0.028
    primary_person_center_smoothing_window_s: float = 1.4
    object_on_samples: int = 2
    object_off_samples: int = 2
    object_unknown_hold_s: float = 9.0
    secondary_unknown_hold_s: float = 9.0

    def __post_init__(self) -> None:
        """Reject malformed debounce and cooldown configuration eagerly."""

        _require_positive_int(self.person_visible_on_samples, field_name="person_visible_on_samples")
        _require_positive_int(self.person_visible_off_samples, field_name="person_visible_off_samples")
        _require_non_negative_float(self.person_visible_unknown_hold_s, field_name="person_visible_unknown_hold_s")
        _require_non_negative_float(self.person_visible_event_cooldown_s, field_name="person_visible_event_cooldown_s")
        _require_non_negative_float(self.person_recently_visible_window_s, field_name="person_recently_visible_window_s")
        _require_non_negative_float(self.person_returned_absence_s, field_name="person_returned_absence_s")
        _require_positive_int(self.looking_toward_device_on_samples, field_name="looking_toward_device_on_samples")
        _require_positive_int(self.looking_toward_device_off_samples, field_name="looking_toward_device_off_samples")
        _require_non_negative_float(
            self.looking_toward_device_unknown_hold_s,
            field_name="looking_toward_device_unknown_hold_s",
        )
        _require_positive_int(self.person_near_device_on_samples, field_name="person_near_device_on_samples")
        _require_positive_int(self.person_near_device_off_samples, field_name="person_near_device_off_samples")
        _require_non_negative_float(
            self.person_near_device_unknown_hold_s,
            field_name="person_near_device_unknown_hold_s",
        )
        _require_positive_int(self.engaged_with_device_on_samples, field_name="engaged_with_device_on_samples")
        _require_positive_int(self.engaged_with_device_off_samples, field_name="engaged_with_device_off_samples")
        _require_non_negative_float(
            self.engaged_with_device_unknown_hold_s,
            field_name="engaged_with_device_unknown_hold_s",
        )
        _require_positive_int(self.showing_intent_on_samples, field_name="showing_intent_on_samples")
        _require_positive_int(self.showing_intent_off_samples, field_name="showing_intent_off_samples")
        _require_non_negative_float(self.showing_intent_unknown_hold_s, field_name="showing_intent_unknown_hold_s")
        _require_non_negative_float(self.showing_intent_event_cooldown_s, field_name="showing_intent_event_cooldown_s")
        _require_positive_int(
            self.hand_or_object_near_camera_on_samples,
            field_name="hand_or_object_near_camera_on_samples",
        )
        _require_positive_int(
            self.hand_or_object_near_camera_off_samples,
            field_name="hand_or_object_near_camera_off_samples",
        )
        _require_non_negative_float(
            self.hand_or_object_near_camera_unknown_hold_s,
            field_name="hand_or_object_near_camera_unknown_hold_s",
        )
        _require_non_negative_float(
            self.hand_or_object_near_camera_event_cooldown_s,
            field_name="hand_or_object_near_camera_event_cooldown_s",
        )
        _require_non_negative_float(self.motion_event_cooldown_s, field_name="motion_event_cooldown_s")
        _require_non_negative_float(self.gesture_event_cooldown_s, field_name="gesture_event_cooldown_s")
        _require_non_negative_float(self.fine_hand_explicit_hold_s, field_name="fine_hand_explicit_hold_s")
        _require_bounded_ratio(
            self.primary_person_center_smoothing_alpha,
            field_name="primary_person_center_smoothing_alpha",
            minimum=_MIN_CENTER_SMOOTHING_ALPHA,
            maximum=_MAX_CENTER_SMOOTHING_ALPHA,
        )
        _require_bounded_ratio(
            self.primary_person_center_deadband,
            field_name="primary_person_center_deadband",
            minimum=_MIN_CENTER_DEADBAND,
            maximum=_MAX_CENTER_DEADBAND,
        )
        _require_bounded_float(
            self.primary_person_center_smoothing_window_s,
            field_name="primary_person_center_smoothing_window_s",
            minimum=_MIN_CENTER_SMOOTHING_WINDOW_S,
            maximum=_MAX_CENTER_SMOOTHING_WINDOW_S,
        )
        _require_positive_int(self.object_on_samples, field_name="object_on_samples")
        _require_positive_int(self.object_off_samples, field_name="object_off_samples")
        _require_non_negative_float(self.object_unknown_hold_s, field_name="object_unknown_hold_s")
        _require_non_negative_float(self.secondary_unknown_hold_s, field_name="secondary_unknown_hold_s")

    @classmethod
    def from_config(cls, config: object) -> "ProactiveCameraSurfaceConfig":
        """Build one cadence-aware camera surface config from Twinr config."""

        interval_s = _coerce_positive_float(getattr(config, "proactive_capture_interval_s", 6.0), default=6.0)
        attention_refresh_s = _coerce_positive_float(
            getattr(config, "display_attention_refresh_interval_s", interval_s),
            default=interval_s,
        )
        unknown_hold_s = max(interval_s + 1.0, interval_s * 1.5)
        cooldown_s = max(interval_s, interval_s * 1.5)
        gesture_cooldown_s = max(0.8, min(2.0, attention_refresh_s * 2.0))
        fine_hand_explicit_hold_s = max(0.2, min(0.8, attention_refresh_s * 0.75))
        # HDMI face-follow must still suppress tiny box wobble, but the old
        # 4x refresh window made person anchors feel visibly delayed on the Pi.
        center_smoothing_window_s = max(
            _MIN_CENTER_SMOOTHING_WINDOW_S,
            min(_MAX_CENTER_SMOOTHING_WINDOW_S, attention_refresh_s * 1.75),
        )
        center_deadband = max(
            0.012,
            min(_MAX_CENTER_DEADBAND, attention_refresh_s * 0.05),
        )
        return cls(
            person_visible_unknown_hold_s=unknown_hold_s,
            person_visible_event_cooldown_s=cooldown_s,
            person_recently_visible_window_s=max(30.0, interval_s * 5.0),
            person_returned_absence_s=_coerce_positive_float(
                getattr(config, "proactive_person_returned_absence_s", 20.0 * 60.0),
                default=20.0 * 60.0,
            ),
            looking_toward_device_unknown_hold_s=unknown_hold_s,
            person_near_device_unknown_hold_s=unknown_hold_s,
            engaged_with_device_unknown_hold_s=unknown_hold_s,
            showing_intent_unknown_hold_s=unknown_hold_s,
            showing_intent_event_cooldown_s=cooldown_s,
            hand_or_object_near_camera_unknown_hold_s=unknown_hold_s,
            hand_or_object_near_camera_event_cooldown_s=cooldown_s,
            motion_event_cooldown_s=cooldown_s,
            gesture_event_cooldown_s=gesture_cooldown_s,
            fine_hand_explicit_hold_s=_coerce_non_negative_float(
                getattr(config, "proactive_local_camera_fine_hand_explicit_hold_s", fine_hand_explicit_hold_s),
                default=fine_hand_explicit_hold_s,
            ),
            primary_person_center_smoothing_alpha=0.76,
            primary_person_center_deadband=center_deadband,
            primary_person_center_smoothing_window_s=center_smoothing_window_s,
            object_unknown_hold_s=unknown_hold_s,
            secondary_unknown_hold_s=unknown_hold_s,
        )


@dataclass(frozen=True, slots=True)
class ProactiveCameraSnapshot:
    """Capture one stabilized camera snapshot for synchronous consumers."""

    camera_online: bool
    camera_online_unknown: bool
    camera_ready: bool
    camera_ready_unknown: bool
    camera_ai_ready: bool
    camera_ai_ready_unknown: bool
    camera_error: str | None
    camera_error_unknown: bool
    last_camera_frame_at: float | None
    last_camera_frame_at_unknown: bool
    last_camera_health_change_at: float | None
    last_camera_health_change_at_unknown: bool
    person_visible: bool
    person_visible_for_s: float
    person_visible_unknown: bool
    person_recently_visible: bool
    person_recently_visible_unknown: bool
    person_count: int
    person_count_unknown: bool
    person_appeared_at: float | None
    person_appeared_at_unknown: bool
    person_disappeared_at: float | None
    person_disappeared_at_unknown: bool
    person_returned_after_absence: bool
    primary_person_zone: SocialPersonZone
    primary_person_zone_unknown: bool
    primary_person_box: SocialSpatialBox | None
    primary_person_box_unknown: bool
    primary_person_center_x: float | None
    primary_person_center_x_unknown: bool
    primary_person_center_y: float | None
    primary_person_center_y_unknown: bool
    looking_toward_device: bool
    looking_toward_device_unknown: bool
    person_near_device: bool
    person_near_device_unknown: bool
    engaged_with_device: bool
    engaged_with_device_unknown: bool
    visual_attention_score: float | None
    visual_attention_score_unknown: bool
    body_pose: SocialBodyPose
    body_pose_unknown: bool
    pose_confidence: float | None
    pose_confidence_unknown: bool
    body_state_changed_at: float | None
    body_state_changed_at_unknown: bool
    motion_state: SocialMotionState
    motion_state_unknown: bool
    motion_confidence: float | None
    motion_confidence_unknown: bool
    motion_state_changed_at: float | None
    motion_state_changed_at_unknown: bool
    smiling: bool
    smiling_unknown: bool
    hand_or_object_near_camera: bool
    hand_or_object_near_camera_for_s: float
    hand_or_object_near_camera_unknown: bool
    showing_intent_likely: bool
    showing_intent_likely_unknown: bool
    showing_intent_started_at: float | None
    showing_intent_started_at_unknown: bool
    gesture_event: SocialGestureEvent
    gesture_event_unknown: bool
    gesture_confidence: float | None
    gesture_confidence_unknown: bool
    fine_hand_gesture: SocialFineHandGesture
    fine_hand_gesture_unknown: bool
    fine_hand_gesture_confidence: float | None
    fine_hand_gesture_confidence_unknown: bool
    objects: tuple[SocialDetectedObject, ...]
    objects_unknown: bool

    @property
    def coarse_arm_gesture(self) -> SocialGestureEvent:
        """Alias the stabilized V1 contract name onto the legacy gesture field."""

        return self.gesture_event

    @property
    def coarse_arm_gesture_unknown(self) -> bool:
        """Alias the stabilized V1 contract unknown flag."""

        return self.gesture_event_unknown

    @property
    def coarse_arm_gesture_confidence(self) -> float | None:
        """Alias the stabilized V1 contract confidence field."""

        return self.gesture_confidence

    @property
    def coarse_arm_gesture_confidence_unknown(self) -> bool:
        """Alias the stabilized V1 contract confidence unknown flag."""

        return self.gesture_confidence_unknown

    @property
    def unknown(self) -> bool:
        """Return whether any camera field is currently running on unknown data."""

        return any(
            (
                self.camera_online_unknown,
                self.camera_ready_unknown,
                self.camera_ai_ready_unknown,
                self.camera_error_unknown,
                self.last_camera_frame_at_unknown,
                self.last_camera_health_change_at_unknown,
                self.person_visible_unknown,
                self.person_recently_visible_unknown,
                self.person_count_unknown,
                self.person_appeared_at_unknown,
                self.person_disappeared_at_unknown,
                self.primary_person_zone_unknown,
                self.primary_person_box_unknown,
                self.primary_person_center_x_unknown,
                self.primary_person_center_y_unknown,
                self.looking_toward_device_unknown,
                self.person_near_device_unknown,
                self.engaged_with_device_unknown,
                self.visual_attention_score_unknown,
                self.body_pose_unknown,
                self.pose_confidence_unknown,
                self.body_state_changed_at_unknown,
                self.motion_state_unknown,
                self.motion_confidence_unknown,
                self.motion_state_changed_at_unknown,
                self.smiling_unknown,
                self.hand_or_object_near_camera_unknown,
                self.showing_intent_likely_unknown,
                self.showing_intent_started_at_unknown,
                self.gesture_event_unknown,
                self.gesture_confidence_unknown,
                self.fine_hand_gesture_unknown,
                self.fine_hand_gesture_confidence_unknown,
                self.objects_unknown,
            )
        )

    def to_automation_facts(self) -> dict[str, object]:
        """Render the snapshot into the runtime automation fact contract."""

        return {
            "camera_online": self.camera_online,
            "camera_online_unknown": self.camera_online_unknown,
            "camera_ready": self.camera_ready,
            "camera_ready_unknown": self.camera_ready_unknown,
            "camera_ai_ready": self.camera_ai_ready,
            "camera_ai_ready_unknown": self.camera_ai_ready_unknown,
            "camera_error": self.camera_error,
            "camera_error_unknown": self.camera_error_unknown,
            "last_camera_frame_at": self.last_camera_frame_at,
            "last_camera_frame_at_unknown": self.last_camera_frame_at_unknown,
            "last_camera_health_change_at": self.last_camera_health_change_at,
            "last_camera_health_change_at_unknown": self.last_camera_health_change_at_unknown,
            "person_visible": self.person_visible,
            "person_visible_for_s": round(self.person_visible_for_s, 3),
            "person_visible_unknown": self.person_visible_unknown,
            "person_recently_visible": self.person_recently_visible,
            "person_recently_visible_unknown": self.person_recently_visible_unknown,
            "person_count": self.person_count,
            "count_persons": self.person_count,
            "person_count_unknown": self.person_count_unknown,
            "person_appeared_at": self.person_appeared_at,
            "person_appeared_at_unknown": self.person_appeared_at_unknown,
            "person_disappeared_at": self.person_disappeared_at,
            "person_disappeared_at_unknown": self.person_disappeared_at_unknown,
            "person_returned_after_absence": self.person_returned_after_absence,
            "primary_person_zone": self.primary_person_zone.value,
            "primary_person_zone_unknown": self.primary_person_zone_unknown,
            "primary_person_box": _serialize_box(self.primary_person_box),
            "primary_person_box_unknown": self.primary_person_box_unknown,
            "primary_person_center_x": self.primary_person_center_x,
            "primary_person_center_x_unknown": self.primary_person_center_x_unknown,
            "primary_person_center_y": self.primary_person_center_y,
            "primary_person_center_y_unknown": self.primary_person_center_y_unknown,
            "looking_toward_device": self.looking_toward_device,
            "looking_toward_device_unknown": self.looking_toward_device_unknown,
            "person_near_device": self.person_near_device,
            "person_near_device_unknown": self.person_near_device_unknown,
            "engaged_with_device": self.engaged_with_device,
            "engaged_with_device_unknown": self.engaged_with_device_unknown,
            "visual_attention_score": self.visual_attention_score,
            "visual_attention_score_unknown": self.visual_attention_score_unknown,
            "body_pose": self.body_pose.value,
            "body_pose_unknown": self.body_pose_unknown,
            "pose_confidence": self.pose_confidence,
            "pose_confidence_unknown": self.pose_confidence_unknown,
            "body_state_changed_at": self.body_state_changed_at,
            "body_state_changed_at_unknown": self.body_state_changed_at_unknown,
            "motion_state": self.motion_state.value,
            "motion_state_unknown": self.motion_state_unknown,
            "motion_confidence": self.motion_confidence,
            "motion_confidence_unknown": self.motion_confidence_unknown,
            "motion_state_changed_at": self.motion_state_changed_at,
            "motion_state_changed_at_unknown": self.motion_state_changed_at_unknown,
            "smiling": self.smiling,
            "smiling_unknown": self.smiling_unknown,
            "hand_or_object_near_camera": self.hand_or_object_near_camera,
            "hand_or_object_near_camera_for_s": round(self.hand_or_object_near_camera_for_s, 3),
            "hand_or_object_near_camera_unknown": self.hand_or_object_near_camera_unknown,
            "showing_intent_likely": self.showing_intent_likely,
            "showing_intent_likely_unknown": self.showing_intent_likely_unknown,
            "showing_intent_started_at": self.showing_intent_started_at,
            "showing_intent_started_at_unknown": self.showing_intent_started_at_unknown,
            "coarse_arm_gesture": self.coarse_arm_gesture.value,
            "coarse_arm_gesture_unknown": self.coarse_arm_gesture_unknown,
            "coarse_arm_gesture_confidence": self.coarse_arm_gesture_confidence,
            "coarse_arm_gesture_confidence_unknown": self.coarse_arm_gesture_confidence_unknown,
            "gesture_event": self.gesture_event.value,
            "gesture_event_unknown": self.gesture_event_unknown,
            "gesture_confidence": self.gesture_confidence,
            "gesture_confidence_unknown": self.gesture_confidence_unknown,
            "fine_hand_gesture": self.fine_hand_gesture.value,
            "fine_hand_gesture_unknown": self.fine_hand_gesture_unknown,
            "fine_hand_gesture_confidence": self.fine_hand_gesture_confidence,
            "fine_hand_gesture_confidence_unknown": self.fine_hand_gesture_confidence_unknown,
            "objects": _serialize_objects(self.objects),
            "objects_unknown": self.objects_unknown,
            "unknown": self.unknown,
        }


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceUpdate:
    """Combine one stabilized snapshot with any newly opened camera events."""

    snapshot: ProactiveCameraSnapshot
    event_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _BooleanSignalView:
    """Describe the latest debounced state of one boolean signal."""

    value: bool
    unknown: bool
    active_for_s: float
    rising_edge: bool


class _DebouncedBooleanSignal:
    """Track one cadence-aware boolean signal with unknown-state handling."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
        event_cooldown_s: float = 0.0,
    ) -> None:
        self.on_samples = _require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = _require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = _require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self.event_cooldown_s = _require_non_negative_float(event_cooldown_s, field_name="event_cooldown_s")
        self._stable_value = False
        self._active_since: float | None = None
        self._last_concrete_at: float | None = None
        self._last_observed_at: float | None = None
        self._pending_value: bool | None = None
        self._pending_count = 0
        self._pending_started_at: float | None = None
        self._last_rising_event_at: float | None = None

    def observe(self, value: bool | None, *, observed_at: float) -> _BooleanSignalView:
        """Observe one new value and return the debounced signal view."""

        now = _coerce_timestamp(observed_at, previous=self._last_observed_at)
        self._last_observed_at = now
        rising_edge = False

        if value is None:
            self._clear_pending()
            if (
                self._last_concrete_at is not None
                and self._stable_value
                and (now - self._last_concrete_at) > self.unknown_hold_s
            ):
                self._stable_value = False
                self._active_since = None
            return _BooleanSignalView(
                value=self._stable_value,
                unknown=True,
                active_for_s=_duration_since(self._active_since, now),
                rising_edge=False,
            )

        self._last_concrete_at = now
        current_value = bool(value)
        if current_value == self._stable_value:
            self._clear_pending()
        else:
            self._advance_pending(current_value, now=now)
            threshold = self.on_samples if current_value else self.off_samples
            if self._pending_count >= threshold:
                self._stable_value = current_value
                if current_value:
                    self._active_since = self._pending_started_at or now
                    rising_edge = self._allow_rising_edge(now)
                else:
                    self._active_since = None
                self._clear_pending()

        if self._stable_value and self._active_since is None:
            self._active_since = now
        return _BooleanSignalView(
            value=self._stable_value,
            unknown=False,
            active_for_s=_duration_since(self._active_since, now),
            rising_edge=rising_edge,
        )

    def _advance_pending(self, value: bool, *, now: float) -> None:
        if self._pending_value is value:
            self._pending_count += 1
            return
        self._pending_value = value
        self._pending_count = 1
        self._pending_started_at = now

    def _clear_pending(self) -> None:
        self._pending_value = None
        self._pending_count = 0
        self._pending_started_at = None

    def _allow_rising_edge(self, now: float) -> bool:
        if self._last_rising_event_at is not None and (now - self._last_rising_event_at) < self.event_cooldown_s:
            return False
        self._last_rising_event_at = now
        return True


@dataclass(frozen=True, slots=True)
class _ObjectTrackerView:
    """Describe the current stable-object surface."""

    objects: tuple[SocialDetectedObject, ...]
    unknown: bool
    rising_objects: tuple[SocialDetectedObject, ...]


class _StableObjectTracker:
    """Track stable object detections across cadence-based observation ticks."""

    def __init__(
        self,
        *,
        on_samples: int,
        off_samples: int,
        unknown_hold_s: float,
    ) -> None:
        self.on_samples = _require_positive_int(on_samples, field_name="on_samples")
        self.off_samples = _require_positive_int(off_samples, field_name="off_samples")
        self.unknown_hold_s = _require_non_negative_float(unknown_hold_s, field_name="unknown_hold_s")
        self._stable: dict[tuple[str, str], SocialDetectedObject] = {}
        self._seen_counts: dict[tuple[str, str], int] = {}
        self._missing_counts: dict[tuple[str, str], int] = {}
        self._last_concrete_at: float | None = None

    def observe(
        self,
        objects: tuple[SocialDetectedObject, ...] | None,
        *,
        observed_at: float,
    ) -> _ObjectTrackerView:
        """Observe one new object list and return the stable surface."""

        now = _coerce_timestamp(observed_at)
        if objects is None:
            if self._last_concrete_at is not None and (now - self._last_concrete_at) > self.unknown_hold_s:
                self._stable.clear()
                self._seen_counts.clear()
                self._missing_counts.clear()
            return _ObjectTrackerView(objects=self._sorted_stable(), unknown=True, rising_objects=())

        self._last_concrete_at = now
        current: dict[tuple[str, str], SocialDetectedObject] = {}
        for item in objects:
            key = _object_key(item)
            previous = current.get(key)
            if previous is None or item.confidence >= previous.confidence:
                current[key] = item

        rising_objects: list[SocialDetectedObject] = []
        for key, item in current.items():
            self._missing_counts[key] = 0
            self._seen_counts[key] = self._seen_counts.get(key, 0) + 1
            if key not in self._stable and self._seen_counts[key] >= self.on_samples:
                stable_item = _stable_object(item)
                self._stable[key] = stable_item
                rising_objects.append(stable_item)
            elif key in self._stable:
                self._stable[key] = _stable_object(item)

        for key in tuple(self._stable):
            if key in current:
                continue
            self._missing_counts[key] = self._missing_counts.get(key, 0) + 1
            if self._missing_counts[key] >= self.off_samples:
                self._stable.pop(key, None)
                self._seen_counts.pop(key, None)
                self._missing_counts.pop(key, None)

        for key in tuple(self._seen_counts):
            if key not in current and key not in self._stable:
                self._seen_counts.pop(key, None)
                self._missing_counts.pop(key, None)

        return _ObjectTrackerView(
            objects=self._sorted_stable(),
            unknown=False,
            rising_objects=tuple(sorted(rising_objects, key=lambda item: (item.label, item.zone.value))),
        )

    def _sorted_stable(self) -> tuple[SocialDetectedObject, ...]:
        return tuple(sorted(self._stable.values(), key=lambda item: (item.label, item.zone.value)))


class ProactiveCameraSurface:
    """Build a stabilized camera snapshot and bounded event surface."""

    @classmethod
    def from_config(cls, config: object) -> "ProactiveCameraSurface":
        return cls(config=ProactiveCameraSurfaceConfig.from_config(config))

    def __init__(self, *, config: ProactiveCameraSurfaceConfig | None = None) -> None:
        self.config = config or ProactiveCameraSurfaceConfig()
        self._person_visible = _DebouncedBooleanSignal(
            on_samples=self.config.person_visible_on_samples,
            off_samples=self.config.person_visible_off_samples,
            unknown_hold_s=self.config.person_visible_unknown_hold_s,
            event_cooldown_s=self.config.person_visible_event_cooldown_s,
        )
        self._looking_toward_device = _DebouncedBooleanSignal(
            on_samples=self.config.looking_toward_device_on_samples,
            off_samples=self.config.looking_toward_device_off_samples,
            unknown_hold_s=self.config.looking_toward_device_unknown_hold_s,
        )
        self._person_near_device = _DebouncedBooleanSignal(
            on_samples=self.config.person_near_device_on_samples,
            off_samples=self.config.person_near_device_off_samples,
            unknown_hold_s=self.config.person_near_device_unknown_hold_s,
        )
        self._engaged_with_device = _DebouncedBooleanSignal(
            on_samples=self.config.engaged_with_device_on_samples,
            off_samples=self.config.engaged_with_device_off_samples,
            unknown_hold_s=self.config.engaged_with_device_unknown_hold_s,
        )
        self._showing_intent = _DebouncedBooleanSignal(
            on_samples=self.config.showing_intent_on_samples,
            off_samples=self.config.showing_intent_off_samples,
            unknown_hold_s=self.config.showing_intent_unknown_hold_s,
            event_cooldown_s=self.config.showing_intent_event_cooldown_s,
        )
        self._hand_near_camera = _DebouncedBooleanSignal(
            on_samples=self.config.hand_or_object_near_camera_on_samples,
            off_samples=self.config.hand_or_object_near_camera_off_samples,
            unknown_hold_s=self.config.hand_or_object_near_camera_unknown_hold_s,
            event_cooldown_s=self.config.hand_or_object_near_camera_event_cooldown_s,
        )
        self._object_tracker = _StableObjectTracker(
            on_samples=self.config.object_on_samples,
            off_samples=self.config.object_off_samples,
            unknown_hold_s=self.config.object_unknown_hold_s,
        )
        self._last_camera_online = False
        self._last_camera_online_at: float | None = None
        self._last_camera_ready = False
        self._last_camera_ready_at: float | None = None
        self._last_camera_ai_ready = False
        self._last_camera_ai_ready_at: float | None = None
        self._last_camera_error: str | None = None
        self._last_camera_error_at: float | None = None
        self._last_camera_frame_at: float | None = None
        self._last_camera_frame_seen_at: float | None = None
        self._last_camera_health_change_at: float | None = None
        self._last_camera_health_change_seen_at: float | None = None
        self._last_body_pose = SocialBodyPose.UNKNOWN
        self._last_body_pose_at: float | None = None
        self._body_state_changed_at: float | None = None
        self._body_state_changed_seen_at: float | None = None
        self._last_pose_confidence: float | None = None
        self._last_pose_confidence_at: float | None = None
        self._last_motion_state = SocialMotionState.UNKNOWN
        self._last_motion_state_at: float | None = None
        self._motion_state_changed_at: float | None = None
        self._motion_state_changed_seen_at: float | None = None
        self._last_motion_confidence: float | None = None
        self._last_motion_confidence_at: float | None = None
        self._last_motion_emitted_at: float | None = None
        self._last_person_count = 0
        self._last_person_count_at: float | None = None
        self._last_primary_person_zone = SocialPersonZone.UNKNOWN
        self._last_primary_person_zone_at: float | None = None
        self._last_primary_person_box: SocialSpatialBox | None = None
        self._last_primary_person_box_at: float | None = None
        self._last_primary_person_center_x: float | None = None
        self._last_primary_person_center_x_at: float | None = None
        self._last_primary_person_center_y: float | None = None
        self._last_primary_person_center_y_at: float | None = None
        self._last_visual_attention_score: float | None = None
        self._last_visual_attention_score_at: float | None = None
        self._last_smiling = False
        self._last_smiling_at: float | None = None
        self._last_gesture_event = SocialGestureEvent.NONE
        self._last_gesture_event_at: float | None = None
        self._last_gesture_confidence: float | None = None
        self._last_gesture_confidence_at: float | None = None
        self._last_gesture_emitted_at: float | None = None
        self._last_gesture_emitted_event = SocialGestureEvent.NONE
        self._last_fine_hand_gesture = SocialFineHandGesture.NONE
        self._last_fine_hand_gesture_at: float | None = None
        self._last_fine_hand_gesture_confidence: float | None = None
        self._last_fine_hand_gesture_confidence_at: float | None = None
        self._last_fine_hand_gesture_emitted_at: float | None = None
        self._last_fine_hand_gesture_emitted_event = SocialFineHandGesture.NONE
        self._last_explicit_fine_hand_gesture = SocialFineHandGesture.NONE
        self._last_explicit_fine_hand_gesture_at: float | None = None
        self._last_explicit_fine_hand_gesture_confidence: float | None = None
        self._has_seen_person = False
        self._last_authoritative_person_visible = False
        self._last_person_seen_at: float | None = None
        self._absence_started_at: float | None = None
        self._person_appeared_at: float | None = None
        self._person_appeared_seen_at: float | None = None
        self._person_disappeared_at: float | None = None
        self._person_disappeared_seen_at: float | None = None
        self._showing_intent_started_at: float | None = None
        self._showing_intent_started_seen_at: float | None = None

    def observe(
        self,
        *,
        inspected: bool,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> ProactiveCameraSurfaceUpdate:
        """Consume one raw observation and return the stabilized camera update."""

        now = _coerce_timestamp(observed_at)
        camera_semantics_authoritative = inspected and _camera_semantics_authoritative(observation)
        person_sample = self._person_visible.observe(
            observation.person_visible if camera_semantics_authoritative else None,
            observed_at=now,
        )
        person_returned_after_absence = self._resolve_person_returned(
            inspected=camera_semantics_authoritative,
            observed_at=now,
            person_visible=person_sample.value,
            person_visible_rising=person_sample.rising_edge,
        )
        person_count, person_count_unknown = self._resolve_person_count(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            person_count=getattr(observation, "person_count", 0),
        )
        primary_person_zone, primary_person_zone_unknown = self._resolve_primary_person_zone(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            primary_person_zone=getattr(observation, "primary_person_zone", SocialPersonZone.UNKNOWN),
        )
        primary_person_box, primary_person_box_unknown = self._resolve_primary_person_box(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            primary_person_box=getattr(observation, "primary_person_box", None),
        )
        primary_person_center_x, primary_person_center_x_unknown = self._resolve_primary_person_center(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_center=getattr(observation, "primary_person_center_x", None),
            box=primary_person_box,
            axis="x",
        )
        primary_person_center_y, primary_person_center_y_unknown = self._resolve_primary_person_center(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_center=getattr(observation, "primary_person_center_y", None),
            box=primary_person_box,
            axis="y",
        )

        looking_sample = self._looking_toward_device.observe(
            (person_sample.value and observation.looking_toward_device) if camera_semantics_authoritative else None,
            observed_at=now,
        )
        person_near_sample = self._person_near_device.observe(
            (
                person_sample.value and bool(getattr(observation, "person_near_device", False))
            ) if camera_semantics_authoritative else None,
            observed_at=now,
        )
        engaged_sample = self._engaged_with_device.observe(
            (
                person_sample.value and bool(getattr(observation, "engaged_with_device", False))
            ) if camera_semantics_authoritative else None,
            observed_at=now,
        )
        hand_sample = self._hand_near_camera.observe(
            (bool(observation.hand_or_object_near_camera) if camera_semantics_authoritative else None),
            observed_at=now,
        )
        showing_sample = self._showing_intent.observe(
            (
                bool(getattr(observation, "showing_intent_likely", False))
                if camera_semantics_authoritative
                else None
            ),
            observed_at=now,
        )
        self._resolve_showing_started_at(
            inspected=camera_semantics_authoritative,
            observed_at=now,
            showing_rising=showing_sample.rising_edge,
        )

        visual_attention_score, visual_attention_score_unknown = self._resolve_visual_attention_score(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_score=getattr(observation, "visual_attention_score", None),
        )
        body_pose, body_pose_unknown = self._resolve_body_pose(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_pose=observation.body_pose,
        )
        pose_confidence, pose_confidence_unknown = self._resolve_pose_confidence(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_confidence=getattr(observation, "pose_confidence", None),
        )
        body_state_changed_at, body_state_changed_at_unknown = self._resolve_body_state_changed_at(
            inspected=camera_semantics_authoritative,
            observed_at=now,
        )
        motion_state, motion_state_unknown, motion_rising = self._resolve_motion_state(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_state=getattr(observation, "motion_state", SocialMotionState.UNKNOWN),
        )
        motion_confidence, motion_confidence_unknown = self._resolve_motion_confidence(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            raw_confidence=getattr(observation, "motion_confidence", None),
        )
        motion_state_changed_at, motion_state_changed_at_unknown = self._resolve_motion_state_changed_at(
            inspected=camera_semantics_authoritative,
            observed_at=now,
        )
        smiling, smiling_unknown = self._resolve_smiling(
            inspected=camera_semantics_authoritative,
            person_visible=person_sample.value,
            observed_at=now,
            smiling=observation.smiling,
        )
        gesture_event, gesture_event_unknown, gesture_confidence, gesture_confidence_unknown, gesture_rising = (
            self._resolve_gesture(
                inspected=camera_semantics_authoritative,
                observed_at=now,
                gesture_event=_coalesce_coarse_gesture_aliases(observation),
                gesture_confidence=getattr(observation, "gesture_confidence", None),
            )
        )
        (
            fine_hand_gesture,
            fine_hand_gesture_unknown,
            fine_hand_gesture_confidence,
            fine_hand_gesture_confidence_unknown,
            fine_hand_gesture_rising,
        ) = self._resolve_fine_hand_gesture(
            inspected=camera_semantics_authoritative,
            observed_at=now,
            fine_hand_gesture=getattr(observation, "fine_hand_gesture", SocialFineHandGesture.NONE),
            fine_hand_gesture_confidence=getattr(observation, "fine_hand_gesture_confidence", None),
        )
        objects_view = self._object_tracker.observe(
            _coerce_detected_objects(getattr(observation, "objects", ())) if camera_semantics_authoritative else None,
            observed_at=now,
        )

        camera_online, camera_online_unknown = self._resolve_secondary_bool(
            inspected=inspected,
            observed_at=now,
            raw_value=getattr(observation, "camera_online", False),
            cache_attr="_last_camera_online",
            cache_seen_attr="_last_camera_online_at",
        )
        camera_ready, camera_ready_unknown = self._resolve_secondary_bool(
            inspected=inspected,
            observed_at=now,
            raw_value=getattr(observation, "camera_ready", False),
            cache_attr="_last_camera_ready",
            cache_seen_attr="_last_camera_ready_at",
        )
        camera_ai_ready, camera_ai_ready_unknown = self._resolve_secondary_bool(
            inspected=inspected,
            observed_at=now,
            raw_value=getattr(observation, "camera_ai_ready", False),
            cache_attr="_last_camera_ai_ready",
            cache_seen_attr="_last_camera_ai_ready_at",
        )
        camera_error, camera_error_unknown = self._resolve_secondary_text(
            inspected=inspected,
            observed_at=now,
            raw_value=getattr(observation, "camera_error", None),
            cache_attr="_last_camera_error",
            cache_seen_attr="_last_camera_error_at",
        )
        last_camera_frame_at, last_camera_frame_at_unknown = self._resolve_secondary_timestamp(
            inspected=inspected,
            observed_at=now,
            raw_value=getattr(observation, "last_camera_frame_at", None),
            cache_attr="_last_camera_frame_at",
            cache_seen_attr="_last_camera_frame_seen_at",
        )
        last_camera_health_change_at, last_camera_health_change_at_unknown = self._resolve_secondary_timestamp(
            inspected=inspected,
            observed_at=now,
            raw_value=getattr(observation, "last_camera_health_change_at", None),
            cache_attr="_last_camera_health_change_at",
            cache_seen_attr="_last_camera_health_change_seen_at",
        )

        person_recently_visible, person_recently_visible_unknown = self._resolve_person_recently_visible(
            observed_at=now,
            person_visible=person_sample.value,
            person_visible_unknown=person_sample.unknown,
        )
        person_appeared_at, person_appeared_at_unknown = self._resolve_transition_timestamp(
            observed_at=now,
            value=self._person_appeared_at,
            last_seen_at=self._person_appeared_seen_at,
        )
        person_disappeared_at, person_disappeared_at_unknown = self._resolve_transition_timestamp(
            observed_at=now,
            value=self._person_disappeared_at,
            last_seen_at=self._person_disappeared_seen_at,
        )
        showing_intent_started_at, showing_intent_started_at_unknown = self._resolve_transition_timestamp(
            observed_at=now,
            value=self._showing_intent_started_at,
            last_seen_at=self._showing_intent_started_seen_at,
        )

        snapshot = ProactiveCameraSnapshot(
            camera_online=camera_online,
            camera_online_unknown=camera_online_unknown,
            camera_ready=camera_ready,
            camera_ready_unknown=camera_ready_unknown,
            camera_ai_ready=camera_ai_ready,
            camera_ai_ready_unknown=camera_ai_ready_unknown,
            camera_error=camera_error,
            camera_error_unknown=camera_error_unknown,
            last_camera_frame_at=last_camera_frame_at,
            last_camera_frame_at_unknown=last_camera_frame_at_unknown,
            last_camera_health_change_at=last_camera_health_change_at,
            last_camera_health_change_at_unknown=last_camera_health_change_at_unknown,
            person_visible=person_sample.value,
            person_visible_for_s=person_sample.active_for_s,
            person_visible_unknown=person_sample.unknown,
            person_recently_visible=person_recently_visible,
            person_recently_visible_unknown=person_recently_visible_unknown,
            person_count=person_count,
            person_count_unknown=person_count_unknown,
            person_appeared_at=person_appeared_at,
            person_appeared_at_unknown=person_appeared_at_unknown,
            person_disappeared_at=person_disappeared_at,
            person_disappeared_at_unknown=person_disappeared_at_unknown,
            person_returned_after_absence=person_returned_after_absence,
            primary_person_zone=primary_person_zone,
            primary_person_zone_unknown=primary_person_zone_unknown,
            primary_person_box=primary_person_box,
            primary_person_box_unknown=primary_person_box_unknown,
            primary_person_center_x=primary_person_center_x,
            primary_person_center_x_unknown=primary_person_center_x_unknown,
            primary_person_center_y=primary_person_center_y,
            primary_person_center_y_unknown=primary_person_center_y_unknown,
            looking_toward_device=looking_sample.value,
            looking_toward_device_unknown=looking_sample.unknown,
            person_near_device=person_near_sample.value,
            person_near_device_unknown=person_near_sample.unknown,
            engaged_with_device=engaged_sample.value,
            engaged_with_device_unknown=engaged_sample.unknown,
            visual_attention_score=visual_attention_score,
            visual_attention_score_unknown=visual_attention_score_unknown,
            body_pose=body_pose,
            body_pose_unknown=body_pose_unknown,
            pose_confidence=pose_confidence,
            pose_confidence_unknown=pose_confidence_unknown,
            body_state_changed_at=body_state_changed_at,
            body_state_changed_at_unknown=body_state_changed_at_unknown,
            motion_state=motion_state,
            motion_state_unknown=motion_state_unknown,
            motion_confidence=motion_confidence,
            motion_confidence_unknown=motion_confidence_unknown,
            motion_state_changed_at=motion_state_changed_at,
            motion_state_changed_at_unknown=motion_state_changed_at_unknown,
            smiling=smiling,
            smiling_unknown=smiling_unknown,
            hand_or_object_near_camera=hand_sample.value,
            hand_or_object_near_camera_for_s=hand_sample.active_for_s,
            hand_or_object_near_camera_unknown=hand_sample.unknown,
            showing_intent_likely=showing_sample.value,
            showing_intent_likely_unknown=showing_sample.unknown,
            showing_intent_started_at=showing_intent_started_at,
            showing_intent_started_at_unknown=showing_intent_started_at_unknown,
            gesture_event=gesture_event,
            gesture_event_unknown=gesture_event_unknown,
            gesture_confidence=gesture_confidence,
            gesture_confidence_unknown=gesture_confidence_unknown,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_unknown=fine_hand_gesture_unknown,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            fine_hand_gesture_confidence_unknown=fine_hand_gesture_confidence_unknown,
            objects=objects_view.objects,
            objects_unknown=objects_view.unknown,
        )

        event_names: list[str] = []
        if person_sample.rising_edge:
            event_names.append(_PERSON_VISIBLE_EVENT)
        if person_returned_after_absence:
            event_names.append(_PERSON_RETURNED_EVENT)
        if hand_sample.rising_edge:
            event_names.append(_HAND_NEAR_EVENT)
        if engaged_sample.rising_edge or looking_sample.rising_edge:
            event_names.append(_ATTENTION_WINDOW_EVENT)
        if showing_sample.rising_edge:
            event_names.append(_SHOWING_INTENT_EVENT)
        if motion_rising:
            event_names.append(_MOTION_EVENT)
        if gesture_rising:
            event_names.append(_GESTURE_EVENT)
            event_names.append(_COARSE_ARM_GESTURE_EVENT)
        if fine_hand_gesture_rising:
            event_names.append(_FINE_HAND_GESTURE_EVENT)
        if objects_view.rising_objects:
            event_names.append(_OBJECT_STABLE_EVENT)
        return ProactiveCameraSurfaceUpdate(snapshot=snapshot, event_names=tuple(event_names))

    def _resolve_person_returned(
        self,
        *,
        inspected: bool,
        observed_at: float,
        person_visible: bool,
        person_visible_rising: bool,
    ) -> bool:
        now = _coerce_timestamp(observed_at)
        person_returned = False
        if inspected:
            if person_visible:
                if (
                    person_visible_rising
                    and self._has_seen_person
                    and self._absence_started_at is not None
                    and (now - self._absence_started_at) >= self.config.person_returned_absence_s
                ):
                    person_returned = True
                self._has_seen_person = True
                self._last_authoritative_person_visible = True
                self._last_person_seen_at = now
                self._absence_started_at = None
                if person_visible_rising:
                    self._person_appeared_at = now
                    self._person_appeared_seen_at = now
            else:
                if self._last_authoritative_person_visible or self._absence_started_at is None:
                    self._absence_started_at = now
                if self._last_authoritative_person_visible:
                    self._person_disappeared_at = now
                    self._person_disappeared_seen_at = now
                self._last_authoritative_person_visible = False
        return person_returned

    def _resolve_person_recently_visible(
        self,
        *,
        observed_at: float,
        person_visible: bool,
        person_visible_unknown: bool,
    ) -> tuple[bool, bool]:
        if person_visible:
            return True, person_visible_unknown
        if self._last_person_seen_at is not None and (observed_at - self._last_person_seen_at) <= self.config.person_recently_visible_window_s:
            return True, person_visible_unknown
        return False, person_visible_unknown

    def _resolve_person_count(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        person_count: object,
    ) -> tuple[int, bool]:
        if inspected:
            if person_visible:
                count = max(1, _coerce_non_negative_int(person_count, default=1))
            else:
                count = 0
            self._last_person_count = count
            self._last_person_count_at = _coerce_timestamp(observed_at)
            return count, False
        return self._hold_secondary(
            value=self._last_person_count,
            last_seen_at=self._last_person_count_at,
            fallback=0,
            observed_at=observed_at,
        )

    def _resolve_primary_person_zone(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_zone: object,
    ) -> tuple[SocialPersonZone, bool]:
        if inspected:
            zone = _coerce_person_zone(primary_person_zone) if person_visible else SocialPersonZone.UNKNOWN
            self._last_primary_person_zone = zone
            self._last_primary_person_zone_at = _coerce_timestamp(observed_at)
            return zone, False
        return self._hold_secondary(
            value=self._last_primary_person_zone,
            last_seen_at=self._last_primary_person_zone_at,
            fallback=SocialPersonZone.UNKNOWN,
            observed_at=observed_at,
        )

    def _resolve_primary_person_box(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        primary_person_box: object,
    ) -> tuple[SocialSpatialBox | None, bool]:
        if inspected:
            box = _coerce_spatial_box(primary_person_box) if person_visible else None
            self._last_primary_person_box = box
            self._last_primary_person_box_at = _coerce_timestamp(observed_at)
            return box, False
        return self._hold_secondary(
            value=self._last_primary_person_box,
            last_seen_at=self._last_primary_person_box_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_primary_person_center(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_center: object,
        box: SocialSpatialBox | None,
        axis: str,
    ) -> tuple[float | None, bool]:
        if inspected:
            if not person_visible:
                value = None
            elif box is not None:
                value = box.center_x if axis == "x" else box.center_y
            else:
                value = _coerce_optional_ratio(raw_center)
            value = self._smooth_primary_person_center(
                axis=axis,
                observed_at=observed_at,
                value=value,
            )
            if axis == "x":
                self._last_primary_person_center_x = value
                self._last_primary_person_center_x_at = _coerce_timestamp(observed_at)
            else:
                self._last_primary_person_center_y = value
                self._last_primary_person_center_y_at = _coerce_timestamp(observed_at)
            return value, False

        if axis == "x":
            return self._hold_secondary(
                value=self._last_primary_person_center_x,
                last_seen_at=self._last_primary_person_center_x_at,
                fallback=None,
                observed_at=observed_at,
            )
        return self._hold_secondary(
            value=self._last_primary_person_center_y,
            last_seen_at=self._last_primary_person_center_y_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _smooth_primary_person_center(
        self,
        *,
        axis: str,
        observed_at: float,
        value: float | None,
    ) -> float | None:
        """Dampen primary-person center jitter without making target switches sluggish."""

        if value is None:
            return None
        previous_value = (
            self._last_primary_person_center_x if axis == "x" else self._last_primary_person_center_y
        )
        previous_at = (
            self._last_primary_person_center_x_at if axis == "x" else self._last_primary_person_center_y_at
        )
        if previous_value is None or previous_at is None:
            return value
        previous_seen_at = _coerce_timestamp(previous_at)
        current_seen_at = _coerce_timestamp(observed_at)
        if previous_seen_at is None or current_seen_at is None:
            return value
        if (current_seen_at - previous_seen_at) > self.config.primary_person_center_smoothing_window_s:
            return value
        delta = value - previous_value
        if abs(delta) <= self.config.primary_person_center_deadband:
            return previous_value
        alpha = self.config.primary_person_center_smoothing_alpha
        smoothed = previous_value + (delta * alpha)
        return max(0.0, min(1.0, smoothed))

    def _resolve_visual_attention_score(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_score: object,
    ) -> tuple[float | None, bool]:
        if inspected:
            value = _coerce_optional_ratio(raw_score) if person_visible else None
            self._last_visual_attention_score = value
            self._last_visual_attention_score_at = _coerce_timestamp(observed_at)
            return value, False
        return self._hold_secondary(
            value=self._last_visual_attention_score,
            last_seen_at=self._last_visual_attention_score_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_body_pose(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_pose: object,
    ) -> tuple[SocialBodyPose, bool]:
        if inspected:
            pose = _coerce_body_pose(raw_pose) if person_visible else SocialBodyPose.UNKNOWN
            if pose != self._last_body_pose:
                self._body_state_changed_at = _coerce_timestamp(observed_at)
                self._body_state_changed_seen_at = _coerce_timestamp(observed_at)
            self._last_body_pose = pose
            self._last_body_pose_at = _coerce_timestamp(observed_at)
            return pose, False
        return self._hold_secondary(
            value=self._last_body_pose,
            last_seen_at=self._last_body_pose_at,
            fallback=SocialBodyPose.UNKNOWN,
            observed_at=observed_at,
        )

    def _resolve_pose_confidence(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_confidence: object,
    ) -> tuple[float | None, bool]:
        if inspected:
            value = _coerce_optional_ratio(raw_confidence) if person_visible else None
            self._last_pose_confidence = value
            self._last_pose_confidence_at = _coerce_timestamp(observed_at)
            return value, False
        return self._hold_secondary(
            value=self._last_pose_confidence,
            last_seen_at=self._last_pose_confidence_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_body_state_changed_at(
        self,
        *,
        inspected: bool,
        observed_at: float,
    ) -> tuple[float | None, bool]:
        if inspected:
            return self._body_state_changed_at, False
        return self._hold_secondary(
            value=self._body_state_changed_at,
            last_seen_at=self._body_state_changed_seen_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_motion_state(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_state: object,
    ) -> tuple[SocialMotionState, bool, bool]:
        """Resolve one held coarse-motion state and whether it changed materially."""

        now = _coerce_timestamp(observed_at)
        if inspected:
            state = _coerce_motion_state(raw_state) if person_visible else SocialMotionState.UNKNOWN
            rising = False
            if state != self._last_motion_state:
                self._motion_state_changed_at = now
                self._motion_state_changed_seen_at = now
                if state not in {SocialMotionState.UNKNOWN, SocialMotionState.STILL}:
                    if (
                        self._last_motion_emitted_at is None
                        or (now - self._last_motion_emitted_at) >= self.config.motion_event_cooldown_s
                    ):
                        self._last_motion_emitted_at = now
                        rising = True
            self._last_motion_state = state
            self._last_motion_state_at = now
            return state, False, rising
        state, unknown = self._hold_secondary(
            value=self._last_motion_state,
            last_seen_at=self._last_motion_state_at,
            fallback=SocialMotionState.UNKNOWN,
            observed_at=observed_at,
        )
        return state, unknown, False

    def _resolve_motion_confidence(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        raw_confidence: object,
    ) -> tuple[float | None, bool]:
        """Resolve one held motion confidence score."""

        if inspected:
            value = _coerce_optional_ratio(raw_confidence) if person_visible else None
            self._last_motion_confidence = value
            self._last_motion_confidence_at = _coerce_timestamp(observed_at)
            return value, False
        return self._hold_secondary(
            value=self._last_motion_confidence,
            last_seen_at=self._last_motion_confidence_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_motion_state_changed_at(
        self,
        *,
        inspected: bool,
        observed_at: float,
    ) -> tuple[float | None, bool]:
        """Resolve the last timestamp where the coarse motion state changed."""

        if inspected:
            return self._motion_state_changed_at, False
        return self._hold_secondary(
            value=self._motion_state_changed_at,
            last_seen_at=self._motion_state_changed_seen_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_smiling(
        self,
        *,
        inspected: bool,
        person_visible: bool,
        observed_at: float,
        smiling: bool,
    ) -> tuple[bool, bool]:
        if inspected:
            self._last_smiling = bool(person_visible and smiling)
            self._last_smiling_at = _coerce_timestamp(observed_at)
            return self._last_smiling, False
        return self._hold_secondary(
            value=self._last_smiling,
            last_seen_at=self._last_smiling_at,
            fallback=False,
            observed_at=observed_at,
        )

    def _resolve_gesture(
        self,
        *,
        inspected: bool,
        observed_at: float,
        gesture_event: object,
        gesture_confidence: object,
    ) -> tuple[SocialGestureEvent, bool, float | None, bool, bool]:
        now = _coerce_timestamp(observed_at)
        if inspected:
            event = _coerce_gesture_event(gesture_event)
            confidence = _coerce_optional_ratio(gesture_confidence)
            self._last_gesture_event = event
            self._last_gesture_event_at = now
            self._last_gesture_confidence = confidence
            self._last_gesture_confidence_at = now
            rising = False
            if event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
                if self._gesture_ready_to_emit(
                    event=event,
                    last_emitted=self._last_gesture_emitted_event,
                    last_emitted_at=self._last_gesture_emitted_at,
                    now=now,
                ):
                    self._last_gesture_emitted_at = now
                    self._last_gesture_emitted_event = event
                    rising = True
            return event, False, confidence, False, rising

        event, event_unknown = self._hold_secondary(
            value=self._last_gesture_event,
            last_seen_at=self._last_gesture_event_at,
            fallback=SocialGestureEvent.UNKNOWN,
            observed_at=observed_at,
        )
        confidence, confidence_unknown = self._hold_secondary(
            value=self._last_gesture_confidence,
            last_seen_at=self._last_gesture_confidence_at,
            fallback=None,
            observed_at=observed_at,
        )
        return event, event_unknown, confidence, confidence_unknown, False

    def _resolve_fine_hand_gesture(
        self,
        *,
        inspected: bool,
        observed_at: float,
        fine_hand_gesture: object,
        fine_hand_gesture_confidence: object,
    ) -> tuple[SocialFineHandGesture, bool, float | None, bool, bool]:
        """Stabilize fine-hand gesture output with the same bounded cadence rules."""

        now = _coerce_timestamp(observed_at)
        if inspected:
            raw_event = _coerce_fine_hand_gesture(fine_hand_gesture)
            raw_confidence = _coerce_optional_ratio(fine_hand_gesture_confidence)
            event, confidence = self._stabilize_fine_hand_gesture(
                event=raw_event,
                confidence=raw_confidence,
                now=now,
            )
            self._last_fine_hand_gesture = event
            self._last_fine_hand_gesture_at = now
            self._last_fine_hand_gesture_confidence = confidence
            self._last_fine_hand_gesture_confidence_at = now
            rising = False
            if event not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
                if self._gesture_ready_to_emit(
                    event=event,
                    last_emitted=self._last_fine_hand_gesture_emitted_event,
                    last_emitted_at=self._last_fine_hand_gesture_emitted_at,
                    now=now,
                ):
                    self._last_fine_hand_gesture_emitted_at = now
                    self._last_fine_hand_gesture_emitted_event = event
                    rising = True
            return event, False, confidence, False, rising

        event, event_unknown = self._hold_secondary(
            value=self._last_fine_hand_gesture,
            last_seen_at=self._last_fine_hand_gesture_at,
            fallback=SocialFineHandGesture.UNKNOWN,
            observed_at=observed_at,
        )
        confidence, confidence_unknown = self._hold_secondary(
            value=self._last_fine_hand_gesture_confidence,
            last_seen_at=self._last_fine_hand_gesture_confidence_at,
            fallback=None,
            observed_at=observed_at,
        )
        return event, event_unknown, confidence, confidence_unknown, False

    def _stabilize_fine_hand_gesture(
        self,
        *,
        event: SocialFineHandGesture,
        confidence: float | None,
        now: float,
    ) -> tuple[SocialFineHandGesture, float | None]:
        """Preserve explicit hand symbols across brief motion/dropout jitter.

        Explicit symbols like thumbs-up or OK-sign are harder for the live Pi
        path than a generic open palm. Keep the last explicit symbol briefly
        when the current frame drops to ``none`` or weakly collapses to
        ``open_palm`` so users do not need to freeze unnaturally.
        """

        if event in _EXPLICIT_FINE_HAND_GESTURES:
            self._last_explicit_fine_hand_gesture = event
            self._last_explicit_fine_hand_gesture_at = now
            self._last_explicit_fine_hand_gesture_confidence = confidence
            return event, confidence

        held_event = self._last_explicit_fine_hand_gesture
        held_at = self._last_explicit_fine_hand_gesture_at
        if held_event not in _EXPLICIT_FINE_HAND_GESTURES or held_at is None:
            return event, confidence
        if (now - held_at) > self.config.fine_hand_explicit_hold_s:
            return event, confidence
        held_confidence = self._last_explicit_fine_hand_gesture_confidence
        current_confidence = 0.0 if confidence is None else confidence
        if event in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}:
            return held_event, held_confidence
        if (
            event == SocialFineHandGesture.OPEN_PALM
            and (held_confidence or 0.0) >= max(0.55, current_confidence + 0.1)
        ):
            return held_event, held_confidence
        return event, confidence

    def _gesture_ready_to_emit(
        self,
        *,
        event: SocialGestureEvent | SocialFineHandGesture,
        last_emitted: SocialGestureEvent | SocialFineHandGesture,
        last_emitted_at: float | None,
        now: float,
    ) -> bool:
        """Allow changed gestures immediately while damping repeated identical ones.

        The local HDMI acknowledgement path refreshes much faster than the slow
        proactive full-inspect cadence. Gesture acknowledgements therefore must
        not inherit multi-second cooldowns between *different* user symbols like
        ``✋`` followed by ``👍``. We still keep a short repeat gate for the same
        gesture so jittery recognizer output does not flash the same emoji on
        every frame.
        """

        if last_emitted_at is None:
            return True
        if event != last_emitted:
            return True
        return (now - last_emitted_at) >= self.config.gesture_event_cooldown_s

    def _resolve_showing_started_at(
        self,
        *,
        inspected: bool,
        observed_at: float,
        showing_rising: bool,
    ) -> None:
        if inspected and showing_rising:
            self._showing_intent_started_at = _coerce_timestamp(observed_at)
            self._showing_intent_started_seen_at = _coerce_timestamp(observed_at)

    def _resolve_transition_timestamp(
        self,
        *,
        observed_at: float,
        value: float | None,
        last_seen_at: float | None,
    ) -> tuple[float | None, bool]:
        return self._hold_secondary(
            value=value,
            last_seen_at=last_seen_at,
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_secondary_bool(
        self,
        *,
        inspected: bool,
        observed_at: float,
        raw_value: object,
        cache_attr: str,
        cache_seen_attr: str,
    ) -> tuple[bool, bool]:
        if inspected:
            value = bool(raw_value)
            setattr(self, cache_attr, value)
            setattr(self, cache_seen_attr, _coerce_timestamp(observed_at))
            return value, False
        return self._hold_secondary(
            value=getattr(self, cache_attr),
            last_seen_at=getattr(self, cache_seen_attr),
            fallback=False,
            observed_at=observed_at,
        )

    def _resolve_secondary_text(
        self,
        *,
        inspected: bool,
        observed_at: float,
        raw_value: object,
        cache_attr: str,
        cache_seen_attr: str,
    ) -> tuple[str | None, bool]:
        if inspected:
            value = _coerce_optional_text(raw_value)
            setattr(self, cache_attr, value)
            setattr(self, cache_seen_attr, _coerce_timestamp(observed_at))
            return value, False
        return self._hold_secondary(
            value=getattr(self, cache_attr),
            last_seen_at=getattr(self, cache_seen_attr),
            fallback=None,
            observed_at=observed_at,
        )

    def _resolve_secondary_timestamp(
        self,
        *,
        inspected: bool,
        observed_at: float,
        raw_value: object,
        cache_attr: str,
        cache_seen_attr: str,
    ) -> tuple[float | None, bool]:
        if inspected:
            value = _coerce_optional_timestamp(raw_value)
            setattr(self, cache_attr, value)
            setattr(self, cache_seen_attr, _coerce_timestamp(observed_at))
            return value, False
        return self._hold_secondary(
            value=getattr(self, cache_attr),
            last_seen_at=getattr(self, cache_seen_attr),
            fallback=None,
            observed_at=observed_at,
        )

    def _hold_secondary(
        self,
        *,
        value: Any,
        last_seen_at: float | None,
        fallback: Any,
        observed_at: float,
    ) -> tuple[Any, bool]:
        if last_seen_at is not None and (_coerce_timestamp(observed_at) - last_seen_at) <= self.config.secondary_unknown_hold_s:
            return value, True
        return fallback, True


def _serialize_box(box: SocialSpatialBox | None) -> dict[str, float] | None:
    """Render one spatial box as a JSON-friendly mapping."""

    if box is None:
        return None
    return {
        "top": round(box.top, 4),
        "left": round(box.left, 4),
        "bottom": round(box.bottom, 4),
        "right": round(box.right, 4),
    }


def _serialize_objects(objects: tuple[SocialDetectedObject, ...]) -> list[dict[str, object]]:
    """Render stable objects into JSON-friendly dictionaries."""

    rendered: list[dict[str, object]] = []
    for item in objects:
        rendered.append(
            {
                "label": item.label,
                "confidence": round(item.confidence, 4),
                "zone": item.zone.value,
                "stable": item.stable,
                "box": _serialize_box(item.box),
            }
        )
    return rendered


def _object_key(item: SocialDetectedObject) -> tuple[str, str]:
    """Return one stable tracker key for an object detection."""

    return (item.label, item.zone.value)


def _stable_object(item: SocialDetectedObject) -> SocialDetectedObject:
    """Return one detection marked as stable for policy consumers."""

    return SocialDetectedObject(
        label=item.label,
        confidence=item.confidence,
        zone=item.zone,
        stable=True,
        box=item.box,
    )


def _coerce_timestamp(value: object, *, previous: float | None = None) -> float:
    """Normalize one timestamp into a monotonic finite float."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0 if previous is None else previous
    number = float(value)
    if not math.isfinite(number):
        number = 0.0 if previous is None else previous
    if previous is not None and number < previous:
        return previous
    return number


def _coerce_optional_timestamp(value: object) -> float | None:
    """Return one finite non-negative timestamp or ``None``."""

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _duration_since(since: float | None, now: float) -> float:
    """Return the elapsed duration since ``since`` with a safe lower bound."""

    if since is None:
        return 0.0
    return max(0.0, now - since)


def _coerce_body_pose(value: object) -> SocialBodyPose:
    """Normalize a raw pose token into the coarse ``SocialBodyPose`` enum."""

    if isinstance(value, SocialBodyPose):
        return value
    if value is None:
        return SocialBodyPose.UNKNOWN
    token = str(value).strip().lower()
    for pose in SocialBodyPose:
        if pose.value == token:
            return pose
    return SocialBodyPose.UNKNOWN


def _coerce_motion_state(value: object) -> SocialMotionState:
    """Normalize a raw motion token into the coarse ``SocialMotionState`` enum."""

    if isinstance(value, SocialMotionState):
        return value
    if value is None:
        return SocialMotionState.UNKNOWN
    token = str(value).strip().lower()
    for state in SocialMotionState:
        if state.value == token:
            return state
    return SocialMotionState.UNKNOWN


def _coerce_person_zone(value: object) -> SocialPersonZone:
    """Normalize a raw zone token into the coarse ``SocialPersonZone`` enum."""

    if isinstance(value, SocialPersonZone):
        return value
    if value is None:
        return SocialPersonZone.UNKNOWN
    token = str(value).strip().lower()
    for zone in SocialPersonZone:
        if zone.value == token:
            return zone
    return SocialPersonZone.UNKNOWN


def _coerce_gesture_event(value: object) -> SocialGestureEvent:
    """Normalize a raw gesture token into the ``SocialGestureEvent`` enum."""

    if isinstance(value, SocialGestureEvent):
        return value
    if value is None:
        return SocialGestureEvent.UNKNOWN
    token = str(value).strip().lower()
    for event in SocialGestureEvent:
        if event.value == token:
            return event
    return SocialGestureEvent.UNKNOWN


def _coalesce_coarse_gesture_aliases(observation: object) -> SocialGestureEvent:
    """Resolve coarse-arm gesture aliases without letting default `none` mask real data.

    `SocialVisionObservation` exposes both `coarse_arm_gesture` and the older
    compatibility field `gesture_event`. The dataclass always has both
    attributes, so a direct `getattr(..., "coarse_arm_gesture", fallback)`
    incorrectly prefers the default `none` value and hides a real gesture
    carried only on `gesture_event`. Prefer the first meaningful non-empty
    value while keeping `coarse_arm_gesture` authoritative when both are set.
    """

    primary = _coerce_gesture_event(getattr(observation, "coarse_arm_gesture", SocialGestureEvent.UNKNOWN))
    if primary not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
        return primary
    secondary = _coerce_gesture_event(getattr(observation, "gesture_event", SocialGestureEvent.UNKNOWN))
    if secondary not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}:
        return secondary
    if primary is not SocialGestureEvent.UNKNOWN:
        return primary
    return secondary


def _coerce_fine_hand_gesture(value: object) -> SocialFineHandGesture:
    """Normalize a raw fine-hand gesture token into the bounded enum."""

    if isinstance(value, SocialFineHandGesture):
        return value
    if value is None:
        return SocialFineHandGesture.UNKNOWN
    token = str(value).strip().lower()
    for event in SocialFineHandGesture:
        if event.value == token:
            return event
    return SocialFineHandGesture.UNKNOWN


def _coerce_optional_ratio(value: object) -> float | None:
    """Clamp one optional numeric value into ``[0.0, 1.0]``."""

    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _coerce_spatial_box(value: object) -> SocialSpatialBox | None:
    """Coerce one box-like payload into ``SocialSpatialBox``."""

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
        return None
    try:
        top, left, bottom, right = (float(item) for item in candidate)
    except (TypeError, ValueError):
        return None
    return SocialSpatialBox(top=top, left=left, bottom=bottom, right=right)


def _coerce_detected_objects(value: object) -> tuple[SocialDetectedObject, ...]:
    """Coerce one list-like payload to a tuple of detected objects."""

    if value is None:
        return ()
    if isinstance(value, tuple) and all(isinstance(item, SocialDetectedObject) for item in value):
        return value
    if not isinstance(value, (tuple, list)):
        return ()
    parsed: list[SocialDetectedObject] = []
    for item in value:
        if isinstance(item, SocialDetectedObject):
            parsed.append(item)
            continue
        if not isinstance(item, dict):
            continue
        parsed.append(
            SocialDetectedObject(
                label=item.get("label", ""),
                confidence=_coerce_optional_ratio(item.get("confidence")) or 0.0,
                zone=_coerce_person_zone(item.get("zone")),
                stable=bool(item.get("stable", False)),
                box=_coerce_spatial_box(item.get("box")),
            )
        )
    return tuple(parsed)


def _coerce_optional_text(value: object) -> str | None:
    """Return one bounded text value or ``None``."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:160]


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


def _camera_health_surface_present(observation: SocialVisionObservation) -> bool:
    """Return whether the observation carries meaningful local camera health."""

    return any(
        (
            bool(getattr(observation, "camera_online", False)),
            bool(getattr(observation, "camera_ready", False)),
            bool(getattr(observation, "camera_ai_ready", False)),
            bool(str(getattr(observation, "camera_error", "") or "").strip()),
            getattr(observation, "last_camera_frame_at", None) is not None,
            getattr(observation, "last_camera_health_change_at", None) is not None,
        )
    )


def _camera_semantics_authoritative(observation: SocialVisionObservation) -> bool:
    """Return whether person/pose/gesture fields describe one real camera frame.

    The local AI-camera path can emit explicit health failures. Those failures
    must make camera semantics unknown, not concrete "no person" facts, so the
    gaze/gesture layers can hold the last stable target through brief runtime
    faults instead of clearing immediately.
    """

    if not _camera_health_surface_present(observation):
        return True
    if not bool(getattr(observation, "camera_online", False)):
        return False
    if not bool(getattr(observation, "camera_ready", False)):
        return False
    if not bool(getattr(observation, "camera_ai_ready", False)):
        return False
    return not bool(str(getattr(observation, "camera_error", "") or "").strip())


def _coerce_positive_float(value: object, *, default: float) -> float:
    """Return a finite positive float, falling back to ``default`` when needed."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    """Return one finite non-negative float, falling back to ``default``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return default
    return number


def _require_positive_int(value: object, *, field_name: str) -> int:
    """Validate and return one positive integer config value."""

    number = _coerce_non_negative_int(value, default=0)
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return number


def _require_non_negative_float(value: object, *, field_name: str) -> float:
    """Validate and return one non-negative float config value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a non-negative float")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field_name} must be a non-negative float")
    return number


def _require_bounded_float(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate and return one bounded float config value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite float")
    number = float(value)
    if not math.isfinite(number) or number < minimum or number > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return number


def _require_bounded_ratio(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    """Validate and return one bounded ratio-like float config value."""

    return _require_bounded_float(
        value,
        field_name=field_name,
        minimum=minimum,
        maximum=maximum,
    )


__all__ = [
    "ProactiveCameraSnapshot",
    "ProactiveCameraSurface",
    "ProactiveCameraSurfaceConfig",
    "ProactiveCameraSurfaceUpdate",
]
