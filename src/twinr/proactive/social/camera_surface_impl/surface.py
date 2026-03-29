# mypy: disable-error-code=assignment
"""Implementation of the proactive camera stabilization surface."""

from __future__ import annotations

from ..engine import SocialBodyPose, SocialFineHandGesture, SocialGestureEvent, SocialMotionState, SocialPersonZone, SocialVisionObservation
from ..perception_stream import attention_stream_authoritative, gesture_stream_authoritative
from .coercion import camera_semantics_authoritative, coalesce_coarse_gesture_aliases, coerce_detected_objects, coerce_timestamp
from .config import ProactiveCameraSurfaceConfig
from .gestures import ProactiveCameraGestureMixin, authoritative_gesture_activation_token
from .models import ProactiveCameraSnapshot, ProactiveCameraSurfaceUpdate
from .presence import ProactiveCameraPresenceMixin
from .signals import DebouncedBooleanSignal, StableObjectTracker
from .validation import coerce_non_negative_int

PERSON_VISIBLE_EVENT = "camera.person_visible"
HAND_NEAR_EVENT = "camera.hand_or_object_near_camera"
PERSON_RETURNED_EVENT = "camera.person_returned"
ATTENTION_WINDOW_EVENT = "camera.attention_window_opened"
SHOWING_INTENT_EVENT = "camera.showing_intent_started"
MOTION_EVENT = "camera.motion_changed"
GESTURE_EVENT = "camera.gesture_detected"
COARSE_ARM_GESTURE_EVENT = "camera.coarse_arm_gesture_detected"
FINE_HAND_GESTURE_EVENT = "camera.fine_hand_gesture_detected"
OBJECT_STABLE_EVENT = "camera.object_detected_stable"


class ProactiveCameraSurfaceImpl(ProactiveCameraPresenceMixin, ProactiveCameraGestureMixin):
    """Build a stabilized camera snapshot and bounded event surface."""

    _coerce_non_negative_int = staticmethod(coerce_non_negative_int)

    @classmethod
    def from_config(cls, config: object) -> "ProactiveCameraSurfaceImpl":
        return cls(config=ProactiveCameraSurfaceConfig.from_config(config))

    def __init__(self, *, config: ProactiveCameraSurfaceConfig | None = None) -> None:
        self.config = config or ProactiveCameraSurfaceConfig()
        self._person_visible = DebouncedBooleanSignal(
            on_samples=self.config.person_visible_on_samples,
            off_samples=self.config.person_visible_off_samples,
            unknown_hold_s=self.config.person_visible_unknown_hold_s,
            event_cooldown_s=self.config.person_visible_event_cooldown_s,
        )
        self._looking_toward_device = DebouncedBooleanSignal(
            on_samples=self.config.looking_toward_device_on_samples,
            off_samples=self.config.looking_toward_device_off_samples,
            unknown_hold_s=self.config.looking_toward_device_unknown_hold_s,
        )
        self._person_near_device = DebouncedBooleanSignal(
            on_samples=self.config.person_near_device_on_samples,
            off_samples=self.config.person_near_device_off_samples,
            unknown_hold_s=self.config.person_near_device_unknown_hold_s,
        )
        self._engaged_with_device = DebouncedBooleanSignal(
            on_samples=self.config.engaged_with_device_on_samples,
            off_samples=self.config.engaged_with_device_off_samples,
            unknown_hold_s=self.config.engaged_with_device_unknown_hold_s,
        )
        self._showing_intent = DebouncedBooleanSignal(
            on_samples=self.config.showing_intent_on_samples,
            off_samples=self.config.showing_intent_off_samples,
            unknown_hold_s=self.config.showing_intent_unknown_hold_s,
            event_cooldown_s=self.config.showing_intent_event_cooldown_s,
        )
        self._hand_near_camera = DebouncedBooleanSignal(
            on_samples=self.config.hand_or_object_near_camera_on_samples,
            off_samples=self.config.hand_or_object_near_camera_off_samples,
            unknown_hold_s=self.config.hand_or_object_near_camera_unknown_hold_s,
            event_cooldown_s=self.config.hand_or_object_near_camera_event_cooldown_s,
        )
        self._object_tracker = StableObjectTracker(
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
        self._last_primary_person_box = None
        self._last_primary_person_box_at: float | None = None
        self._last_visible_persons = ()
        self._last_visible_persons_at: float | None = None
        self._last_primary_person_center_x: float | None = None
        self._last_primary_person_center_x_at: float | None = None
        self._last_primary_person_center_y: float | None = None
        self._last_primary_person_center_y_at: float | None = None
        self._last_visual_attention_score: float | None = None
        self._last_visual_attention_score_at: float | None = None
        self._last_looking_signal_state: str | None = None
        self._last_looking_signal_state_at: float | None = None
        self._last_looking_signal_source: str | None = None
        self._last_looking_signal_source_at: float | None = None
        self._last_smiling = False
        self._last_smiling_at: float | None = None
        self._last_gesture_event = SocialGestureEvent.NONE
        self._last_gesture_event_at: float | None = None
        self._last_gesture_confidence: float | None = None
        self._last_gesture_confidence_at: float | None = None
        self._last_gesture_emitted_at: float | None = None
        self._last_gesture_emitted_event = SocialGestureEvent.NONE
        self._last_gesture_emitted_token: int | None = None
        self._last_fine_hand_gesture = SocialFineHandGesture.NONE
        self._last_fine_hand_gesture_at: float | None = None
        self._last_fine_hand_gesture_confidence: float | None = None
        self._last_fine_hand_gesture_confidence_at: float | None = None
        self._last_fine_hand_gesture_emitted_at: float | None = None
        self._last_fine_hand_gesture_emitted_event = SocialFineHandGesture.NONE
        self._last_fine_hand_gesture_emitted_token: int | None = None
        self._last_explicit_fine_hand_gesture = SocialFineHandGesture.NONE
        self._last_explicit_fine_hand_gesture_at: float | None = None
        self._last_explicit_fine_hand_gesture_confidence: float | None = None
        self._pending_explicit_fine_hand_gesture = SocialFineHandGesture.NONE
        self._pending_explicit_fine_hand_gesture_count = 0
        self._pending_explicit_fine_hand_gesture_confidence: float | None = None
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

        now = coerce_timestamp(observed_at)
        camera_semantics_ok = inspected and camera_semantics_authoritative(observation)
        attention_temporal_authoritative = bool(
            camera_semantics_ok and attention_stream_authoritative(observation)
        )
        gesture_temporal_authoritative = bool(
            camera_semantics_ok and gesture_stream_authoritative(observation)
        )
        gesture_activation_token = authoritative_gesture_activation_token(observation)
        person_sample = self._person_visible.observe(
            observation.person_visible if camera_semantics_ok else None,
            observed_at=now,
        )
        person_returned_after_absence = self._resolve_person_returned(
            inspected=camera_semantics_ok,
            observed_at=now,
            person_visible=person_sample.value,
            person_visible_rising=person_sample.rising_edge,
        )
        person_count, person_count_unknown = self._resolve_person_count(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            person_count=getattr(observation, "person_count", 0),
        )
        primary_person_zone, primary_person_zone_unknown = self._resolve_primary_person_zone(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            primary_person_zone=getattr(observation, "primary_person_zone", SocialPersonZone.UNKNOWN),
        )
        primary_person_box, primary_person_box_unknown = self._resolve_primary_person_box(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            primary_person_box=getattr(observation, "primary_person_box", None),
        )
        visible_persons, visible_persons_unknown = self._resolve_visible_persons(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            visible_persons=getattr(observation, "visible_persons", ()),
            primary_person_box=primary_person_box,
            primary_person_zone=primary_person_zone,
        )
        primary_person_center_x, primary_person_center_x_unknown = self._resolve_primary_person_center(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_center=getattr(observation, "primary_person_center_x", None),
            box=primary_person_box,
            axis="x",
        )
        primary_person_center_y, primary_person_center_y_unknown = self._resolve_primary_person_center(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_center=getattr(observation, "primary_person_center_y", None),
            box=primary_person_box,
            axis="y",
        )

        looking_sample_input = (
            person_sample.value and observation.looking_toward_device
        ) if camera_semantics_ok else None
        if attention_temporal_authoritative:
            looking_sample = self._looking_toward_device.observe_authoritative(
                looking_sample_input,
                observed_at=now,
            )
        else:
            looking_sample = self._looking_toward_device.observe(
                looking_sample_input,
                observed_at=now,
            )
        looking_signal_state, looking_signal_state_unknown = self._resolve_secondary_text(
            inspected=camera_semantics_ok,
            observed_at=now,
            raw_value=(
                getattr(observation, "looking_signal_state", None)
                if person_sample.value and looking_sample.value
                else None
            ),
            cache_attr="_last_looking_signal_state",
            cache_seen_attr="_last_looking_signal_state_at",
        )
        looking_signal_source, looking_signal_source_unknown = self._resolve_secondary_text(
            inspected=camera_semantics_ok,
            observed_at=now,
            raw_value=(
                getattr(observation, "looking_signal_source", None)
                if person_sample.value and looking_sample.value
                else None
            ),
            cache_attr="_last_looking_signal_source",
            cache_seen_attr="_last_looking_signal_source_at",
        )
        person_near_sample_input = (
            person_sample.value and bool(getattr(observation, "person_near_device", False))
        ) if camera_semantics_ok else None
        if attention_temporal_authoritative:
            person_near_sample = self._person_near_device.observe_authoritative(
                person_near_sample_input,
                observed_at=now,
            )
        else:
            person_near_sample = self._person_near_device.observe(
                person_near_sample_input,
                observed_at=now,
            )
        engaged_sample_input = (
            person_sample.value and bool(getattr(observation, "engaged_with_device", False))
        ) if camera_semantics_ok else None
        if attention_temporal_authoritative:
            engaged_sample = self._engaged_with_device.observe_authoritative(
                engaged_sample_input,
                observed_at=now,
            )
        else:
            engaged_sample = self._engaged_with_device.observe(
                engaged_sample_input,
                observed_at=now,
            )
        hand_sample_input = bool(observation.hand_or_object_near_camera) if camera_semantics_ok else None
        if attention_temporal_authoritative or gesture_temporal_authoritative:
            hand_sample = self._hand_near_camera.observe_authoritative(
                hand_sample_input,
                observed_at=now,
            )
        else:
            hand_sample = self._hand_near_camera.observe(
                hand_sample_input,
                observed_at=now,
            )
        showing_sample_input = (
            bool(getattr(observation, "showing_intent_likely", False))
            if camera_semantics_ok
            else None
        )
        if attention_temporal_authoritative or gesture_temporal_authoritative:
            showing_sample = self._showing_intent.observe_authoritative(
                showing_sample_input,
                observed_at=now,
            )
        else:
            showing_sample = self._showing_intent.observe(
                showing_sample_input,
                observed_at=now,
            )
        self._resolve_showing_started_at(
            inspected=camera_semantics_ok,
            observed_at=now,
            showing_rising=showing_sample.rising_edge,
        )

        visual_attention_score, visual_attention_score_unknown = self._resolve_visual_attention_score(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_score=getattr(observation, "visual_attention_score", None),
        )
        body_pose, body_pose_unknown = self._resolve_body_pose(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_pose=observation.body_pose,
        )
        pose_confidence, pose_confidence_unknown = self._resolve_pose_confidence(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_confidence=getattr(observation, "pose_confidence", None),
        )
        body_state_changed_at, body_state_changed_at_unknown = self._resolve_body_state_changed_at(
            inspected=camera_semantics_ok,
            observed_at=now,
        )
        motion_state, motion_state_unknown, motion_rising = self._resolve_motion_state(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_state=getattr(observation, "motion_state", SocialMotionState.UNKNOWN),
        )
        motion_confidence, motion_confidence_unknown = self._resolve_motion_confidence(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            raw_confidence=getattr(observation, "motion_confidence", None),
        )
        motion_state_changed_at, motion_state_changed_at_unknown = self._resolve_motion_state_changed_at(
            inspected=camera_semantics_ok,
            observed_at=now,
        )
        smiling, smiling_unknown = self._resolve_smiling(
            inspected=camera_semantics_ok,
            person_visible=person_sample.value,
            observed_at=now,
            smiling=observation.smiling,
        )
        gesture_event, gesture_event_unknown, gesture_confidence, gesture_confidence_unknown, gesture_rising = (
            self._resolve_gesture(
                inspected=camera_semantics_ok,
                observed_at=now,
                gesture_event=coalesce_coarse_gesture_aliases(observation),
                gesture_confidence=getattr(observation, "gesture_confidence", None),
                temporal_authoritative=gesture_temporal_authoritative,
                activation_token=gesture_activation_token,
            )
        )
        (
            fine_hand_gesture,
            fine_hand_gesture_unknown,
            fine_hand_gesture_confidence,
            fine_hand_gesture_confidence_unknown,
            fine_hand_gesture_rising,
        ) = self._resolve_fine_hand_gesture(
            inspected=camera_semantics_ok,
            observed_at=now,
            fine_hand_gesture=getattr(observation, "fine_hand_gesture", SocialFineHandGesture.NONE),
            fine_hand_gesture_confidence=getattr(observation, "fine_hand_gesture_confidence", None),
            temporal_authoritative=gesture_temporal_authoritative,
            activation_token=gesture_activation_token,
        )
        objects_view = self._object_tracker.observe(
            coerce_detected_objects(getattr(observation, "objects", ())) if camera_semantics_ok else None,
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
            visible_persons=visible_persons,
            visible_persons_unknown=visible_persons_unknown,
            primary_person_center_x=primary_person_center_x,
            primary_person_center_x_unknown=primary_person_center_x_unknown,
            primary_person_center_y=primary_person_center_y,
            primary_person_center_y_unknown=primary_person_center_y_unknown,
            looking_toward_device=looking_sample.value,
            looking_toward_device_unknown=looking_sample.unknown,
            looking_signal_state=looking_signal_state,
            looking_signal_state_unknown=looking_signal_state_unknown,
            looking_signal_source=looking_signal_source,
            looking_signal_source_unknown=looking_signal_source_unknown,
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
            event_names.append(PERSON_VISIBLE_EVENT)
        if person_returned_after_absence:
            event_names.append(PERSON_RETURNED_EVENT)
        if hand_sample.rising_edge:
            event_names.append(HAND_NEAR_EVENT)
        if engaged_sample.rising_edge or looking_sample.rising_edge:
            event_names.append(ATTENTION_WINDOW_EVENT)
        if showing_sample.rising_edge:
            event_names.append(SHOWING_INTENT_EVENT)
        if motion_rising:
            event_names.append(MOTION_EVENT)
        if gesture_rising:
            event_names.append(GESTURE_EVENT)
            event_names.append(COARSE_ARM_GESTURE_EVENT)
        if fine_hand_gesture_rising:
            event_names.append(FINE_HAND_GESTURE_EVENT)
        if objects_view.rising_objects:
            event_names.append(OBJECT_STABLE_EVENT)
        return ProactiveCameraSurfaceUpdate(snapshot=snapshot, event_names=tuple(event_names))
