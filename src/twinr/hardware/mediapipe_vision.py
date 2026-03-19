"""Expose the stable MediaPipe camera API over the decomposed camera stack.

This module is intentionally thin. The actual MediaPipe runtime and classifiers
live under ``twinr.hardware.camera_ai`` so Twinr keeps the historic import path
without keeping MediaPipe logic in one monolith.
"""

from __future__ import annotations

from .camera_ai.config import (
    DEFAULT_MEDIAPIPE_GESTURE_MODEL_URL,
    DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL,
    DEFAULT_MEDIAPIPE_POSE_MODEL_URL,
    MediaPipeVisionConfig,
)
from .camera_ai.fine_hand_gestures import (
    prefer_gesture_choice as _prefer_gesture_choice,
    resolve_fine_hand_gesture as _resolve_fine_hand_gesture,
)
from .camera_ai.mediapipe_pipeline import (
    MediaPipeVisionPipeline,
    MediaPipeVisionResult,
    extract_sparse_keypoints as _extract_sparse_keypoints,
)
from .camera_ai.temporal_gestures import (
    TemporalPoseGestureClassifier,
    classify_temporal_gesture as _classify_temporal_gesture,
)

__all__ = [
    "DEFAULT_MEDIAPIPE_GESTURE_MODEL_URL",
    "DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL",
    "DEFAULT_MEDIAPIPE_POSE_MODEL_URL",
    "MediaPipeVisionConfig",
    "MediaPipeVisionPipeline",
    "MediaPipeVisionResult",
    "TemporalPoseGestureClassifier",
]
