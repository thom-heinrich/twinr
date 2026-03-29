"""Fuse recent rich camera semantics into the fast HDMI attention lane.

The dedicated HDMI attention refresh intentionally skips expensive pose and hand
inference to keep face-follow responsive. This helper lets the display path
reuse recent richer observations from the slower full-inspection path and the
gesture lane, so short fast-path dropouts do not blank the header pills or snap
the face back to center.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from twinr.agent.base_agent.config import TwinrConfig

from ..social.engine import (
    SocialBodyPose,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialVisionObservation,
)
from .display_attention import resolve_display_attention_refresh_interval


@dataclass(frozen=True, slots=True)
class DisplayAttentionCameraFusionConfig:
    """Store bounded freshness windows for display-only camera fusion."""

    dropout_hold_s: float
    gesture_semantic_hold_s: float
    pose_semantic_hold_s: float
    anchor_match_delta_x: float = 0.2
    anchor_match_delta_y: float = 0.24

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAttentionCameraFusionConfig":
        """Derive display-only fusion windows from the current runtime cadence."""

        refresh_interval_s = resolve_display_attention_refresh_interval(config)
        if refresh_interval_s is None:
            try:
                refresh_interval_s = float(getattr(config, "proactive_capture_interval_s", 6.0) or 6.0)
            except (TypeError, ValueError):
                refresh_interval_s = 0.35
        try:
            proactive_capture_interval_s = float(getattr(config, "proactive_capture_interval_s", 6.0) or 6.0)
        except (TypeError, ValueError):
            proactive_capture_interval_s = 6.0
        refresh_interval_s = max(0.2, float(refresh_interval_s))
        proactive_capture_interval_s = max(refresh_interval_s, float(proactive_capture_interval_s))
        return cls(
            dropout_hold_s=max(0.75, min(1.5, refresh_interval_s * 3.0)),
            gesture_semantic_hold_s=max(0.9, min(2.5, refresh_interval_s * 4.0)),
            # Full pose/social observations arrive much slower than the HDMI
            # fast path, so keep those richer semantics alive until the next
            # normal proactive inspection should have had a chance to refresh.
            pose_semantic_hold_s=max(2.0, min(8.0, proactive_capture_interval_s + refresh_interval_s)),
        )


@dataclass(frozen=True, slots=True)
class DisplayAttentionCameraFusionResult:
    """Return one fused attention observation plus bounded debug metadata."""

    observation: SocialVisionObservation
    debug_details: dict[str, object]


@dataclass(frozen=True, slots=True)
class _StoredObservation:
    """Remember one recent source observation with its capture time."""

    observed_at: float
    observation: SocialVisionObservation
    source: str


class DisplayAttentionCameraFusion:
    """Fuse fast attention observations with recent pose and gesture semantics."""

    def __init__(self, *, config: DisplayAttentionCameraFusionConfig) -> None:
        self.config = config
        self._last_full_observation: _StoredObservation | None = None
        self._last_gesture_observation: _StoredObservation | None = None
        self._last_fused_visible_observation: _StoredObservation | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "DisplayAttentionCameraFusion":
        """Build one display-only fusion helper from runtime config."""

        return cls(config=DisplayAttentionCameraFusionConfig.from_config(config))

    def remember_full(self, *, observed_at: float, observation: SocialVisionObservation) -> None:
        """Remember one richer full-inspection camera observation."""

        self._last_full_observation = _StoredObservation(
            observed_at=float(observed_at),
            observation=observation,
            source="full_observe",
        )

    def remember_gesture(self, *, observed_at: float, observation: SocialVisionObservation) -> None:
        """Remember one recent gesture-lane camera observation."""

        self._last_gesture_observation = _StoredObservation(
            observed_at=float(observed_at),
            observation=observation,
            source="gesture_refresh",
        )

    def fuse_attention(
        self,
        *,
        observed_at: float,
        observation: SocialVisionObservation,
    ) -> DisplayAttentionCameraFusionResult:
        """Fuse recent richer semantics into one fast attention observation."""

        checked_at = float(observed_at)
        full_age_s = _age_s(checked_at, self._last_full_observation)
        gesture_age_s = _age_s(checked_at, self._last_gesture_observation)
        fused_age_s = _age_s(checked_at, self._last_fused_visible_observation)
        debug_details: dict[str, object] = {
            "fast_person_visible": observation.person_visible,
            "fast_person_count": observation.person_count,
            "full_age_s": _round_seconds(full_age_s),
            "gesture_age_s": _round_seconds(gesture_age_s),
            "last_fused_age_s": _round_seconds(fused_age_s),
            "used_pose_source": None,
            "used_hand_source": None,
            "dropout_hold_source": None,
        }

        fused = observation
        if fused.person_visible:
            pose_source = self._best_pose_source(
                observed_at=checked_at,
                current=fused,
            )
            if pose_source is not None:
                fused = _apply_pose_semantics(fused, pose_source.observation)
                debug_details["used_pose_source"] = pose_source.source

            hand_source = self._best_hand_source(
                observed_at=checked_at,
                current=fused,
            )
            if hand_source is not None:
                fused = _apply_hand_semantics(fused, hand_source.observation)
                debug_details["used_hand_source"] = hand_source.source
        else:
            held_visible = self._fresh_visible_source(
                self._last_fused_visible_observation,
                observed_at=checked_at,
                max_age_s=self.config.dropout_hold_s,
            )
            if held_visible is not None:
                fused = _merge_health_fields(current=observation, held=held_visible.observation)
                debug_details["dropout_hold_source"] = held_visible.source

        if fused.person_visible:
            self._last_fused_visible_observation = _StoredObservation(
                observed_at=checked_at,
                observation=fused,
                source="attention_fused",
            )

        debug_details.update(
            {
                "result_person_visible": fused.person_visible,
                "result_looking_toward_device": fused.looking_toward_device,
                "result_looking_signal_state": fused.looking_signal_state,
                "result_looking_signal_source": fused.looking_signal_source,
                "result_hand_or_object_near_camera": fused.hand_or_object_near_camera,
                "result_showing_intent_likely": fused.showing_intent_likely,
                "result_body_pose": fused.body_pose.value,
                "result_motion_state": fused.motion_state.value,
            }
        )
        return DisplayAttentionCameraFusionResult(
            observation=fused,
            debug_details=debug_details,
        )

    def _best_pose_source(
        self,
        *,
        observed_at: float,
        current: SocialVisionObservation,
    ) -> _StoredObservation | None:
        """Return the freshest compatible source that carries pose semantics."""

        candidates = (
            self._compatible_visible_source(
                self._last_full_observation,
                observed_at=observed_at,
                current=current,
                max_age_s=self.config.pose_semantic_hold_s,
                require_semantics=_has_pose_semantics,
            ),
            self._compatible_visible_source(
                self._last_fused_visible_observation,
                observed_at=observed_at,
                current=current,
                max_age_s=self.config.dropout_hold_s,
                require_semantics=_has_pose_semantics,
            ),
        )
        return _freshest(candidates)

    def _best_hand_source(
        self,
        *,
        observed_at: float,
        current: SocialVisionObservation,
    ) -> _StoredObservation | None:
        """Return the freshest compatible source that carries hand semantics."""

        candidates = (
            self._compatible_visible_source(
                self._last_gesture_observation,
                observed_at=observed_at,
                current=current,
                max_age_s=self.config.gesture_semantic_hold_s,
                require_semantics=_has_hand_semantics,
            ),
            self._compatible_visible_source(
                self._last_full_observation,
                observed_at=observed_at,
                current=current,
                max_age_s=self.config.pose_semantic_hold_s,
                require_semantics=_has_hand_semantics,
            ),
            self._compatible_visible_source(
                self._last_fused_visible_observation,
                observed_at=observed_at,
                current=current,
                max_age_s=self.config.dropout_hold_s,
                require_semantics=_has_hand_semantics,
            ),
        )
        return _freshest(candidates)

    def _compatible_visible_source(
        self,
        stored: _StoredObservation | None,
        *,
        observed_at: float,
        current: SocialVisionObservation,
        max_age_s: float,
        require_semantics,
    ) -> _StoredObservation | None:
        """Return one recent compatible visible source or ``None``."""

        candidate = self._fresh_visible_source(stored, observed_at=observed_at, max_age_s=max_age_s)
        if candidate is None:
            return None
        if not require_semantics(candidate.observation):
            return None
        if not _anchors_compatible(
            current=current,
            candidate=candidate.observation,
            config=self.config,
        ):
            return None
        return candidate

    def _fresh_visible_source(
        self,
        stored: _StoredObservation | None,
        *,
        observed_at: float,
        max_age_s: float,
    ) -> _StoredObservation | None:
        """Return one recent source only while it still shows a visible person."""

        if stored is None or not stored.observation.person_visible:
            return None
        age_s = observed_at - float(stored.observed_at)
        if age_s < 0.0 or age_s > max_age_s:
            return None
        return stored


def _freshest(candidates: tuple[_StoredObservation | None, ...]) -> _StoredObservation | None:
    """Return the newest source from a small fixed candidate set."""

    available = [candidate for candidate in candidates if candidate is not None]
    if not available:
        return None
    return max(available, key=lambda item: item.observed_at)


def _age_s(observed_at: float, stored: _StoredObservation | None) -> float | None:
    """Return the age of one stored observation."""

    if stored is None:
        return None
    age_s = observed_at - float(stored.observed_at)
    if age_s < 0.0:
        return 0.0
    return age_s


def _round_seconds(value: float | None) -> float | None:
    """Round one optional age to bounded debug precision."""

    if value is None:
        return None
    return round(float(value), 3)


def _has_pose_semantics(observation: SocialVisionObservation) -> bool:
    """Return whether one source carries richer pose/motion semantics."""

    return any(
        (
            observation.body_pose != SocialBodyPose.UNKNOWN,
            observation.motion_state != SocialMotionState.UNKNOWN,
            observation.person_recently_visible is not None,
            observation.person_appeared_at is not None,
            observation.person_disappeared_at is not None,
        )
    )


def _has_hand_semantics(observation: SocialVisionObservation) -> bool:
    """Return whether one source carries richer hand/intent semantics."""

    return any(
        (
            observation.hand_or_object_near_camera,
            observation.showing_intent_likely is True,
            observation.fine_hand_gesture not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN},
            observation.gesture_event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN},
        )
    )


def _anchors_compatible(
    *,
    current: SocialVisionObservation,
    candidate: SocialVisionObservation,
    config: DisplayAttentionCameraFusionConfig,
) -> bool:
    """Return whether one recent source still targets the same visible person."""

    if not current.person_visible or not candidate.person_visible:
        return False
    current_center_x = current.primary_person_center_x
    candidate_center_x = candidate.primary_person_center_x
    current_center_y = current.primary_person_center_y
    candidate_center_y = candidate.primary_person_center_y
    if current_center_x is not None and candidate_center_x is not None:
        if abs(float(current_center_x) - float(candidate_center_x)) <= config.anchor_match_delta_x:
            if current_center_y is None or candidate_center_y is None:
                return True
            if abs(float(current_center_y) - float(candidate_center_y)) <= config.anchor_match_delta_y:
                return True
    if (
        current.primary_person_zone != SocialPersonZone.UNKNOWN
        and current.primary_person_zone == candidate.primary_person_zone
    ):
        return True
    if (
        current.person_count <= 1
        and candidate.person_count <= 1
        and current.primary_person_zone == SocialPersonZone.UNKNOWN
        and candidate.primary_person_zone == SocialPersonZone.UNKNOWN
        and current_center_x is None
        and candidate_center_x is None
    ):
        return True
    return False


def _apply_pose_semantics(
    current: SocialVisionObservation,
    candidate: SocialVisionObservation,
) -> SocialVisionObservation:
    """Overlay recent non-authoritative pose semantics onto one fast observation."""

    updates: dict[str, object] = {}
    if candidate.body_pose != SocialBodyPose.UNKNOWN and current.body_pose == SocialBodyPose.UNKNOWN:
        updates["body_pose"] = candidate.body_pose
        updates["pose_confidence"] = candidate.pose_confidence
    if candidate.motion_state != SocialMotionState.UNKNOWN and current.motion_state == SocialMotionState.UNKNOWN:
        updates["motion_state"] = candidate.motion_state
        updates["motion_confidence"] = candidate.motion_confidence
    if current.person_recently_visible is None and candidate.person_recently_visible is not None:
        updates["person_recently_visible"] = candidate.person_recently_visible
    if current.person_appeared_at is None and candidate.person_appeared_at is not None:
        updates["person_appeared_at"] = candidate.person_appeared_at
    if current.person_disappeared_at is None and candidate.person_disappeared_at is not None:
        updates["person_disappeared_at"] = candidate.person_disappeared_at
    if not updates:
        return current
    return replace(current, **updates)


def _apply_hand_semantics(
    current: SocialVisionObservation,
    candidate: SocialVisionObservation,
) -> SocialVisionObservation:
    """Overlay recent hand/intent semantics onto one fast observation."""

    updates: dict[str, object] = {}
    if candidate.hand_or_object_near_camera and not current.hand_or_object_near_camera:
        updates["hand_or_object_near_camera"] = True
    if candidate.showing_intent_likely is True and current.showing_intent_likely is not True:
        updates["showing_intent_likely"] = True
        if current.showing_intent_started_at is None and candidate.showing_intent_started_at is not None:
            updates["showing_intent_started_at"] = candidate.showing_intent_started_at
    if candidate.fine_hand_gesture not in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN} and (
        current.fine_hand_gesture in {SocialFineHandGesture.NONE, SocialFineHandGesture.UNKNOWN}
    ):
        updates["fine_hand_gesture"] = candidate.fine_hand_gesture
        updates["fine_hand_gesture_confidence"] = candidate.fine_hand_gesture_confidence
    if candidate.gesture_event not in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN} and (
        current.gesture_event in {SocialGestureEvent.NONE, SocialGestureEvent.UNKNOWN}
    ):
        updates["gesture_event"] = candidate.gesture_event
        updates["coarse_arm_gesture"] = candidate.gesture_event
        updates["gesture_confidence"] = candidate.gesture_confidence
    if not updates:
        return current
    return replace(current, **updates)


def _merge_health_fields(
    *,
    current: SocialVisionObservation,
    held: SocialVisionObservation,
) -> SocialVisionObservation:
    """Keep the held visible person while preserving current camera health."""

    return replace(
        held,
        camera_online=current.camera_online,
        camera_ready=current.camera_ready,
        camera_ai_ready=current.camera_ai_ready,
        camera_error=current.camera_error,
        last_camera_frame_at=(
            current.last_camera_frame_at
            if current.last_camera_frame_at is not None
            else held.last_camera_frame_at
        ),
        last_camera_health_change_at=(
            current.last_camera_health_change_at
            if current.last_camera_health_change_at is not None
            else held.last_camera_health_change_at
        ),
    )


__all__ = [
    "DisplayAttentionCameraFusion",
    "DisplayAttentionCameraFusionConfig",
    "DisplayAttentionCameraFusionResult",
]
