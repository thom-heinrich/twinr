"""Expose the stable public IMX500 camera API over the decomposed camera stack.

This module is intentionally thin. The actual implementation lives under
``twinr.hardware.camera_ai`` so the public import path stays stable while the
camera stack remains separated by concern.
"""

from __future__ import annotations

from .camera_ai.adapter import LocalAICameraAdapter
from .camera_ai.config import AICameraAdapterConfig
from .camera_ai.models import (
    AICameraBodyPose,
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraMotionState,
    AICameraObjectDetection,
    AICameraObservation,
    AICameraZone,
)
from .camera_ai.motion import infer_motion_state as _infer_motion_state
from .camera_ai.pose_classification import (
    classify_body_pose as _classify_body_pose,
    classify_gesture as _classify_gesture,
)
from .camera_ai.pose_features import (
    attention_score as _attention_score,
    hand_near_camera as _hand_near_camera,
    parse_keypoints as _parse_keypoints,
    strong_keypoint_count as _strong_keypoint_count,
    support_pose_confidence as _support_pose_confidence,
    visible_joint as _visible_joint,
)
from .camera_ai.pose_selection import rank_pose_candidates as _rank_pose_candidates

__all__ = [
    "AICameraAdapterConfig",
    "AICameraBodyPose",
    "AICameraBox",
    "AICameraFineHandGesture",
    "AICameraGestureEvent",
    "AICameraMotionState",
    "AICameraObjectDetection",
    "AICameraObservation",
    "AICameraZone",
    "LocalAICameraAdapter",
]
