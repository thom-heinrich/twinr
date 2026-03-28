"""Data containers used by the local AI-camera adapter implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    AICameraBodyPose,
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
)

@dataclass(frozen=True, slots=True)
class PoseResult:
    """Describe one parsed pose frame."""

    body_pose: AICameraBodyPose
    pose_confidence: float | None
    looking_toward_device: bool | None
    visual_attention_score: float | None
    hand_near_camera: bool
    showing_intent_likely: bool | None
    gesture_event: AICameraGestureEvent
    gesture_confidence: float | None
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    sparse_keypoints: dict[int, tuple[float, float, float]] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class GesturePersonTargets:
    """Describe the bounded person boxes that the gesture lane should trust."""

    primary_person_box: AICameraBox | None
    visible_person_boxes: tuple[AICameraBox, ...] = ()
    person_count: int = 0
    source: str = "imx500"
    face_anchor_state: str = "disabled"
    face_anchor_count: int = 0
