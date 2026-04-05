# CHANGELOG: 2026-03-29
# BUG-1: Enforced monotonic observation timestamps; stale frames are dropped and duplicate timestamps are micro-stepped.
# BUG-2: Missing sparse fields no longer collapse to False/0/() and no longer flap debounced state or stable object tracking.
# BUG-3: Partial observations no longer crash on direct access to body_pose/smiling.
# SEC-1: Sanitized/capped free-form text and bounded visible-person/object fan-out to reduce log, memory, and CPU amplification on Raspberry Pi 4.
# IMP-1: Added observe() reentrancy/thread-safety guard for live-stream callback pipelines.
# IMP-2: Added stronger sparse-field fallbacks (person_count derivation, authoritative field-presence gating) for modern multi-rate edge vision stacks.
"""Implementation of the proactive camera stabilization surface."""

from __future__ import annotations

from itertools import islice
from math import isfinite
from threading import RLock

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

_MISSING = object()


class ProactiveCameraSurfaceImpl(ProactiveCameraPresenceMixin, ProactiveCameraGestureMixin):
    """Build a stabilized camera snapshot and bounded event surface."""

    _MAX_VISIBLE_PERSONS = 16
    _MAX_OBJECTS = 32
    _MAX_SIGNAL_TEXT_LEN = 96
    _MAX_CAMERA_ERROR_LEN = 512
    _TIMESTAMP_EPSILON_S = 1e-6

    _coerce_non_negative_int = staticmethod(coerce_non_negative_int)

    @classmethod
    def from_config(cls, config: object) -> "ProactiveCameraSurfaceImpl":
        return cls(config=ProactiveCameraSurfaceConfig.from_config(config))

    def __init__(self, *, config: ProactiveCameraSurfaceConfig | None = None) -> None:
        self.config = config or ProactiveCameraSurfaceConfig()
        self._lock = RLock()
        self._person_visible = DebouncedBooleanSignal(
            on_samples=self.config.person_visible_on_samples,
            off_samples=self.config.person_visible_off_samples,
            unknown_hold_s=self.config.person_visible_unknown_hold_s,
            event_cooldown_s=self.config.person_visible_event_cooldown_s,
            on_min_s=0.0,
            off_min_s=0.0,
        )
        self._looking_toward_device = DebouncedBooleanSignal(
            on_samples=self.config.looking_toward_device_on_samples,
            off_samples=self.config.looking_toward_device_off_samples,
            unknown_hold_s=self.config.looking_toward_device_unknown_hold_s,
            on_min_s=0.0,
            off_min_s=0.0,
        )
        self._person_near_device = DebouncedBooleanSignal(
            on_samples=self.config.person_near_device_on_samples,
            off_samples=self.config.person_near_device_off_samples,
            unknown_hold_s=self.config.person_near_device_unknown_hold_s,
            on_min_s=0.0,
            off_min_s=0.0,
        )
        self._engaged_with_device = DebouncedBooleanSignal(
            on_samples=self.config.engaged_with_device_on_samples,
            off_samples=self.config.engaged_with_device_off_samples,
            unknown_hold_s=self.config.engaged_with_device_unknown_hold_s,
            on_min_s=0.0,
            off_min_s=0.0,
        )
        self._showing_intent = DebouncedBooleanSignal(
            on_samples=self.config.showing_intent_on_samples,
            off_samples=self.config.showing_intent_off_samples,
            unknown_hold_s=self.config.showing_intent_unknown_hold_s,
            event_cooldown_s=self.config.showing_intent_event_cooldown_s,
            on_min_s=0.0,
            off_min_s=0.0,
        )
        self._hand_near_camera = DebouncedBooleanSignal(
            on_samples=self.config.hand_or_object_near_camera_on_samples,
            off_samples=self.config.hand_or_object_near_camera_off_samples,
            unknown_hold_s=self.config.hand_or_object_near_camera_unknown_hold_s,
            event_cooldown_s=self.config.hand_or_object_near_camera_event_cooldown_s,
            on_min_s=0.0,
            off_min_s=0.0,
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
        self._last_input_observed_at: float | None = None
        self._last_effective_observed_at: float | None = None
        self._last_surface_snapshot: ProactiveCameraSnapshot | None = None
        self._stale_observation_count = 0
        self._same_timestamp_observation_count = 0

    @staticmethod
    def _attr_present(observation: object, attr_name: str) -> bool:
        return getattr(observation, attr_name, _MISSING) is not _MISSING

    @staticmethod
    def _coerce_optional_bool(value: object) -> bool | None:
        if value is None or value is _MISSING:
            return None
        if isinstance(value, (str, bytes)):
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                if not isfinite(float(value)):
                    return None
            except (TypeError, ValueError):
                return None
        return bool(value)

    @classmethod
    def _read_optional_bool(cls, observation: object, attr_name: str) -> bool | None:
        return cls._coerce_optional_bool(getattr(observation, attr_name, _MISSING))

    @staticmethod
    def _coerce_optional_finite_float(value: object) -> float | None:
        if value is None or value is _MISSING:
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if not isfinite(coerced):
            return None
        return coerced

    @classmethod
    def _coerce_optional_unit_float(cls, value: object) -> float | None:
        coerced = cls._coerce_optional_finite_float(value)
        if coerced is None:
            return None
        if coerced < 0.0:
            return 0.0
        if coerced > 1.0:
            return 1.0
        return coerced

    @classmethod
    def _sanitize_optional_text(cls, value: object, *, max_len: int) -> str | None:
        if value is None or value is _MISSING:
            return None
        sanitized = "".join(
            ch if ch.isprintable() and ch not in "\r\n\t\x00" else " "
            for ch in str(value)
        )
        sanitized = " ".join(sanitized.split())
        if not sanitized:
            return None
        if len(sanitized) > max_len:
            sanitized = sanitized[: max_len - 1].rstrip() + "…"
        return sanitized

    @staticmethod
    def _limit_sequence(value: object, *, max_items: int) -> tuple[object, ...]:
        if value is None or value is _MISSING:
            return ()
        if isinstance(value, (str, bytes)):
            return ()
        if isinstance(value, tuple):
            return value[:max_items]
        if isinstance(value, list):
            return tuple(value[:max_items])
        try:
            return tuple(islice(iter(value), max_items))
        except TypeError:
            return ()

    @classmethod
    def _read_limited_sequence(
        cls,
        observation: object,
        attr_name: str,
        *,
        max_items: int,
    ) -> tuple[object, ...]:
        return cls._limit_sequence(
            getattr(observation, attr_name, _MISSING),
            max_items=max_items,
        )

    def _normalize_observed_at(self, observed_at: float) -> tuple[float, bool]:
        now = coerce_timestamp(observed_at)
        if self._last_input_observed_at is None or self._last_effective_observed_at is None:
            self._last_input_observed_at = now
            self._last_effective_observed_at = now
            return now, False
        if now < self._last_input_observed_at:
            self._stale_observation_count += 1
            return self._last_effective_observed_at, True
        if now == self._last_input_observed_at:
            self._same_timestamp_observation_count += 1
            now = self._last_effective_observed_at + self._TIMESTAMP_EPSILON_S
        else:
            self._last_input_observed_at = now
        self._last_effective_observed_at = now
        return now, False

    def _derive_person_count_input(
        self,
        *,
        person_visible: bool,
        raw_person_count: object,
        raw_visible_persons: tuple[object, ...],
        raw_primary_person_box: object,
        raw_primary_person_zone: object,
    ) -> int:
        if raw_person_count is not _MISSING and raw_person_count is not None:
            # Keep the sparse-field fallback chain alive when upstream camera
            # payloads send an unusable person_count value.
            person_count = self._coerce_non_negative_int(raw_person_count, default=-1)
            if person_count >= 0:
                return person_count
        if raw_visible_persons:
            return len(raw_visible_persons)
        if person_visible and (
            raw_primary_person_box is not None
            or raw_primary_person_zone not in (_MISSING, None, SocialPersonZone.UNKNOWN)
        ):
            return 1
        return 1 if person_visible else 0

    def observe(
        self,
        *,
        inspected: bool,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> ProactiveCameraSurfaceUpdate:
        """Consume one raw observation and return the stabilized camera update."""

        with self._lock:
            now, stale_observation = self._normalize_observed_at(observed_at)
            if stale_observation and self._last_surface_snapshot is not None:
                return ProactiveCameraSurfaceUpdate(
                    snapshot=self._last_surface_snapshot,
                    event_names=(),
                )

            camera_semantics_ok = inspected and camera_semantics_authoritative(observation)
            attention_temporal_authoritative = bool(
                camera_semantics_ok and attention_stream_authoritative(observation)
            )
            gesture_temporal_authoritative = bool(
                camera_semantics_ok and gesture_stream_authoritative(observation)
            )
            gesture_activation_token = authoritative_gesture_activation_token(observation)

            raw_person_visible = self._read_optional_bool(observation, "person_visible")
            raw_looking_toward_device = self._read_optional_bool(observation, "looking_toward_device")
            raw_person_near_device = self._read_optional_bool(observation, "person_near_device")
            raw_engaged_with_device = self._read_optional_bool(observation, "engaged_with_device")
            raw_hand_near_camera = self._read_optional_bool(observation, "hand_or_object_near_camera")
            raw_showing_intent = self._read_optional_bool(observation, "showing_intent_likely")
            raw_smiling = self._read_optional_bool(observation, "smiling")
            raw_person_count = getattr(observation, "person_count", _MISSING)
            raw_primary_person_zone = getattr(observation, "primary_person_zone", _MISSING)
            raw_primary_person_box = (
                getattr(observation, "primary_person_box", None)
                if self._attr_present(observation, "primary_person_box")
                else None
            )
            raw_visible_persons = self._read_limited_sequence(
                observation,
                "visible_persons",
                max_items=self._MAX_VISIBLE_PERSONS,
            )
            raw_primary_person_center_x = self._coerce_optional_finite_float(
                getattr(observation, "primary_person_center_x", None)
            )
            raw_primary_person_center_y = self._coerce_optional_finite_float(
                getattr(observation, "primary_person_center_y", None)
            )
            raw_visual_attention_score = self._coerce_optional_finite_float(
                getattr(observation, "visual_attention_score", None)
            )
            raw_body_pose = getattr(observation, "body_pose", SocialBodyPose.UNKNOWN)
            raw_pose_confidence = self._coerce_optional_unit_float(
                getattr(observation, "pose_confidence", None)
            )
            raw_motion_state = getattr(observation, "motion_state", SocialMotionState.UNKNOWN)
            raw_motion_confidence = self._coerce_optional_unit_float(
                getattr(observation, "motion_confidence", None)
            )
            raw_gesture_confidence = self._coerce_optional_unit_float(
                getattr(observation, "gesture_confidence", None)
            )
            raw_fine_hand_gesture = getattr(
                observation,
                "fine_hand_gesture",
                SocialFineHandGesture.NONE,
            )
            raw_fine_hand_gesture_confidence = self._coerce_optional_unit_float(
                getattr(observation, "fine_hand_gesture_confidence", None)
            )
            raw_looking_signal_state = self._sanitize_optional_text(
                getattr(observation, "looking_signal_state", None),
                max_len=self._MAX_SIGNAL_TEXT_LEN,
            )
            raw_looking_signal_source = self._sanitize_optional_text(
                getattr(observation, "looking_signal_source", None),
                max_len=self._MAX_SIGNAL_TEXT_LEN,
            )
            raw_camera_online = self._read_optional_bool(observation, "camera_online")
            raw_camera_ready = self._read_optional_bool(observation, "camera_ready")
            raw_camera_ai_ready = self._read_optional_bool(observation, "camera_ai_ready")
            raw_camera_error = self._sanitize_optional_text(
                getattr(observation, "camera_error", None),
                max_len=self._MAX_CAMERA_ERROR_LEN,
            )
            raw_last_camera_frame_at = self._coerce_optional_finite_float(
                getattr(observation, "last_camera_frame_at", None)
            )
            raw_last_camera_health_change_at = self._coerce_optional_finite_float(
                getattr(observation, "last_camera_health_change_at", None)
            )

            person_sample = self._person_visible.observe(
                raw_person_visible if camera_semantics_ok else None,
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
                person_count=self._derive_person_count_input(
                    person_visible=person_sample.value,
                    raw_person_count=raw_person_count,
                    raw_visible_persons=raw_visible_persons,
                    raw_primary_person_box=raw_primary_person_box,
                    raw_primary_person_zone=raw_primary_person_zone,
                ),
            )
            primary_person_zone, primary_person_zone_unknown = self._resolve_primary_person_zone(
                inspected=camera_semantics_ok and (
                    not person_sample.value or self._attr_present(observation, "primary_person_zone")
                ),
                person_visible=person_sample.value,
                observed_at=now,
                primary_person_zone=(
                    raw_primary_person_zone
                    if raw_primary_person_zone is not _MISSING
                    else SocialPersonZone.UNKNOWN
                ),
            )
            primary_person_box, primary_person_box_unknown = self._resolve_primary_person_box(
                inspected=camera_semantics_ok and (
                    not person_sample.value or self._attr_present(observation, "primary_person_box")
                ),
                person_visible=person_sample.value,
                observed_at=now,
                primary_person_box=raw_primary_person_box,
            )
            visible_persons, visible_persons_unknown = self._resolve_visible_persons(
                inspected=camera_semantics_ok and (
                    not person_sample.value or self._attr_present(observation, "visible_persons")
                ),
                person_visible=person_sample.value,
                observed_at=now,
                visible_persons=raw_visible_persons,
                primary_person_box=primary_person_box,
                primary_person_zone=primary_person_zone,
            )
            primary_person_center_x, primary_person_center_x_unknown = self._resolve_primary_person_center(
                inspected=camera_semantics_ok and (
                    not person_sample.value
                    or self._attr_present(observation, "primary_person_center_x")
                    or raw_primary_person_box is not None
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_center=raw_primary_person_center_x,
                box=primary_person_box,
                axis="x",
            )
            primary_person_center_y, primary_person_center_y_unknown = self._resolve_primary_person_center(
                inspected=camera_semantics_ok and (
                    not person_sample.value
                    or self._attr_present(observation, "primary_person_center_y")
                    or raw_primary_person_box is not None
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_center=raw_primary_person_center_y,
                box=primary_person_box,
                axis="y",
            )

            if camera_semantics_ok:
                looking_sample_input = (
                    False if not person_sample.value else raw_looking_toward_device
                )
            else:
                looking_sample_input = None
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
                inspected=camera_semantics_ok and (
                    not (person_sample.value and looking_sample.value)
                    or self._attr_present(observation, "looking_signal_state")
                ),
                observed_at=now,
                raw_value=(
                    raw_looking_signal_state
                    if person_sample.value and looking_sample.value
                    else None
                ),
                cache_attr="_last_looking_signal_state",
                cache_seen_attr="_last_looking_signal_state_at",
            )
            looking_signal_source, looking_signal_source_unknown = self._resolve_secondary_text(
                inspected=camera_semantics_ok and (
                    not (person_sample.value and looking_sample.value)
                    or self._attr_present(observation, "looking_signal_source")
                ),
                observed_at=now,
                raw_value=(
                    raw_looking_signal_source
                    if person_sample.value and looking_sample.value
                    else None
                ),
                cache_attr="_last_looking_signal_source",
                cache_seen_attr="_last_looking_signal_source_at",
            )

            if camera_semantics_ok:
                person_near_sample_input = (
                    False if not person_sample.value else raw_person_near_device
                )
            else:
                person_near_sample_input = None
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

            if camera_semantics_ok:
                engaged_sample_input = (
                    False if not person_sample.value else raw_engaged_with_device
                )
            else:
                engaged_sample_input = None
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

            hand_sample_input = raw_hand_near_camera if camera_semantics_ok else None
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

            showing_sample_input = raw_showing_intent if camera_semantics_ok else None
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
                inspected=camera_semantics_ok and (
                    not person_sample.value or raw_visual_attention_score is not None
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_score=raw_visual_attention_score,
            )
            body_pose, body_pose_unknown = self._resolve_body_pose(
                inspected=camera_semantics_ok and (
                    not person_sample.value or self._attr_present(observation, "body_pose")
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_pose=raw_body_pose,
            )
            pose_confidence, pose_confidence_unknown = self._resolve_pose_confidence(
                inspected=camera_semantics_ok and (
                    not person_sample.value or raw_pose_confidence is not None
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_confidence=raw_pose_confidence,
            )
            body_state_changed_at, body_state_changed_at_unknown = self._resolve_body_state_changed_at(
                inspected=camera_semantics_ok,
                observed_at=now,
            )
            motion_state, motion_state_unknown, motion_rising = self._resolve_motion_state(
                inspected=camera_semantics_ok and (
                    not person_sample.value or self._attr_present(observation, "motion_state")
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_state=raw_motion_state,
            )
            motion_confidence, motion_confidence_unknown = self._resolve_motion_confidence(
                inspected=camera_semantics_ok and (
                    not person_sample.value or raw_motion_confidence is not None
                ),
                person_visible=person_sample.value,
                observed_at=now,
                raw_confidence=raw_motion_confidence,
            )
            motion_state_changed_at, motion_state_changed_at_unknown = self._resolve_motion_state_changed_at(
                inspected=camera_semantics_ok,
                observed_at=now,
            )
            smiling, smiling_unknown = self._resolve_smiling(
                inspected=camera_semantics_ok and (
                    not person_sample.value or self._attr_present(observation, "smiling")
                ),
                person_visible=person_sample.value,
                observed_at=now,
                smiling=bool(raw_smiling) if raw_smiling is not None else False,
            )
            gesture_event, gesture_event_unknown, gesture_confidence, gesture_confidence_unknown, gesture_rising = (
                self._resolve_gesture(
                    inspected=camera_semantics_ok,
                    observed_at=now,
                    gesture_event=coalesce_coarse_gesture_aliases(observation),
                    gesture_confidence=raw_gesture_confidence,
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
                inspected=camera_semantics_ok and self._attr_present(observation, "fine_hand_gesture"),
                observed_at=now,
                fine_hand_gesture=raw_fine_hand_gesture,
                fine_hand_gesture_confidence=raw_fine_hand_gesture_confidence,
                temporal_authoritative=gesture_temporal_authoritative,
                activation_token=gesture_activation_token,
            )

            objects_input = None
            if camera_semantics_ok and self._attr_present(observation, "objects"):
                objects_input = self._limit_sequence(
                    coerce_detected_objects(getattr(observation, "objects", ())),
                    max_items=self._MAX_OBJECTS,
                )
            objects_view = self._object_tracker.observe(
                objects_input,
                observed_at=now,
            )

            camera_online, camera_online_unknown = self._resolve_secondary_bool(
                inspected=inspected and raw_camera_online is not None,
                observed_at=now,
                raw_value=bool(raw_camera_online) if raw_camera_online is not None else False,
                cache_attr="_last_camera_online",
                cache_seen_attr="_last_camera_online_at",
            )
            camera_ready, camera_ready_unknown = self._resolve_secondary_bool(
                inspected=inspected and raw_camera_ready is not None,
                observed_at=now,
                raw_value=bool(raw_camera_ready) if raw_camera_ready is not None else False,
                cache_attr="_last_camera_ready",
                cache_seen_attr="_last_camera_ready_at",
            )
            camera_ai_ready, camera_ai_ready_unknown = self._resolve_secondary_bool(
                inspected=inspected and raw_camera_ai_ready is not None,
                observed_at=now,
                raw_value=bool(raw_camera_ai_ready) if raw_camera_ai_ready is not None else False,
                cache_attr="_last_camera_ai_ready",
                cache_seen_attr="_last_camera_ai_ready_at",
            )
            camera_error, camera_error_unknown = self._resolve_secondary_text(
                inspected=inspected and self._attr_present(observation, "camera_error"),
                observed_at=now,
                raw_value=raw_camera_error,
                cache_attr="_last_camera_error",
                cache_seen_attr="_last_camera_error_at",
            )
            last_camera_frame_at, last_camera_frame_at_unknown = self._resolve_secondary_timestamp(
                inspected=inspected and raw_last_camera_frame_at is not None,
                observed_at=now,
                raw_value=raw_last_camera_frame_at,
                cache_attr="_last_camera_frame_at",
                cache_seen_attr="_last_camera_frame_seen_at",
            )
            last_camera_health_change_at, last_camera_health_change_at_unknown = self._resolve_secondary_timestamp(
                inspected=inspected and raw_last_camera_health_change_at is not None,
                observed_at=now,
                raw_value=raw_last_camera_health_change_at,
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

            if len(event_names) > 1:
                event_names = list(dict.fromkeys(event_names))

            self._last_surface_snapshot = snapshot
            return ProactiveCameraSurfaceUpdate(
                snapshot=snapshot,
                event_names=tuple(event_names),
            )
