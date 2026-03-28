"""Public snapshot models produced by the camera surface."""

from __future__ import annotations

from dataclasses import dataclass

from ..engine import (
    SocialBodyPose,
    SocialDetectedObject,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisiblePerson,
)
from .coercion import serialize_box, serialize_objects, serialize_visible_persons


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
    visible_persons: tuple[SocialVisiblePerson, ...] = ()
    visible_persons_unknown: bool = False
    looking_signal_state: str | None = None
    looking_signal_state_unknown: bool = False
    looking_signal_source: str | None = None
    looking_signal_source_unknown: bool = False

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
                self.visible_persons_unknown,
                self.primary_person_center_x_unknown,
                self.primary_person_center_y_unknown,
                self.looking_toward_device_unknown,
                self.looking_signal_state_unknown,
                self.looking_signal_source_unknown,
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
            "primary_person_box": serialize_box(self.primary_person_box),
            "primary_person_box_unknown": self.primary_person_box_unknown,
            "visible_persons": serialize_visible_persons(self.visible_persons),
            "visible_persons_unknown": self.visible_persons_unknown,
            "primary_person_center_x": self.primary_person_center_x,
            "primary_person_center_x_unknown": self.primary_person_center_x_unknown,
            "primary_person_center_y": self.primary_person_center_y,
            "primary_person_center_y_unknown": self.primary_person_center_y_unknown,
            "looking_toward_device": self.looking_toward_device,
            "looking_toward_device_unknown": self.looking_toward_device_unknown,
            "looking_signal_state": self.looking_signal_state,
            "looking_signal_state_unknown": self.looking_signal_state_unknown,
            "looking_signal_source": self.looking_signal_source,
            "looking_signal_source_unknown": self.looking_signal_source_unknown,
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
            "objects": serialize_objects(self.objects),
            "objects_unknown": self.objects_unknown,
            "unknown": self.unknown,
        }


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceUpdate:
    """Combine one stabilized snapshot with any newly opened camera events."""

    snapshot: ProactiveCameraSnapshot
    event_names: tuple[str, ...] = ()
