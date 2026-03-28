# mypy: disable-error-code="assignment"
"""Primary implementation class for the decomposed AI-camera adapter."""

from __future__ import annotations

from threading import Lock
from typing import Any
import time

from twinr.agent.base_agent.config import TwinrConfig

from ..config import AICameraAdapterConfig
from ..face_anchors import OpenCVFaceAnchorDetector
from ..gesture_candidate_capture import GestureCandidateCaptureStore
from ..imx500_runtime import IMX500RuntimeSessionManager
from ..live_gesture_pipeline import LiveGesturePipeline
from ..mediapipe_pipeline import MediaPipeVisionPipeline
from ..models import AICameraMotionState
from .attention import AICameraAdapterAttentionMixin
from .common import LOGGER
from .composition import AICameraAdapterCompositionMixin
from .gesture import AICameraAdapterGestureMixin
from .observe import AICameraAdapterObserveMixin
from .pose import AICameraAdapterPoseMixin
from .runtime import AICameraAdapterRuntimeMixin
from .state import AICameraAdapterStateMixin
from .types import PoseResult

logger = LOGGER


class LocalAICameraAdapter(
    AICameraAdapterObserveMixin,
    AICameraAdapterAttentionMixin,
    AICameraAdapterGestureMixin,
    AICameraAdapterPoseMixin,
    AICameraAdapterCompositionMixin,
    AICameraAdapterRuntimeMixin,
    AICameraAdapterStateMixin,
):
    """Provide one bounded local-first IMX500 observation surface."""

    def __init__(
        self,
        *,
        config: AICameraAdapterConfig | None = None,
        face_anchor_detector: object | None = None,
        clock: Any = time.time,
        sleep_fn: Any = time.sleep,
        monotonic_clock: Any = time.monotonic,
    ) -> None:
        """Initialize one bounded adapter with lazy runtime dependencies."""

        self.config = config or AICameraAdapterConfig()
        self._clock = clock
        self._sleep = sleep_fn
        self._monotonic_clock = monotonic_clock  # AUDIT-FIX(#4): Use monotonic time for internal duration math.
        self._lock = Lock()
        self._health_lock = Lock()  # AUDIT-FIX(#1): Health/frame metadata can be updated by timeout callers outside the main capture lock.
        self._runtime_manager = IMX500RuntimeSessionManager(config=self.config, sleep_fn=self._sleep)
        self._gesture_candidate_capture = GestureCandidateCaptureStore.from_config(
            self.config,
            clock=self._clock,
        )
        self._last_frame_at: float | None = None
        self._last_health_change_at: float | None = None
        self._last_health_signature: tuple[bool, bool, bool, str | None] | None = None
        self._last_pose_at: float | None = None
        self._last_pose_monotonic: float | None = None  # AUDIT-FIX(#4): Cache freshness must be independent of wall-clock jumps.
        self._last_pose_result: PoseResult | None = None
        self._last_pose_box_metrics: dict[str, float] | None = None  # AUDIT-FIX(#3): Reuse cached pose only for the same tracked person.
        self._last_pose_hint_keypoints: dict[int, tuple[float, float, float]] = {}
        self._last_pose_hint_confidence: float | None = None
        self._last_pose_hint_monotonic: float | None = None
        self._last_pose_hint_box_metrics: dict[str, float] | None = None
        self._mediapipe_pipeline: MediaPipeVisionPipeline | None = None
        self._live_gesture_pipeline: LiveGesturePipeline | None = None
        self._last_motion_box = None
        self._last_motion_person_count = 0
        self._last_motion_at: float | None = None
        self._last_motion_monotonic: float | None = None  # AUDIT-FIX(#4): Motion deltas must use monotonic time.
        self._last_motion_state = AICameraMotionState.UNKNOWN
        self._last_motion_confidence: float | None = None
        self._face_anchor_detector = face_anchor_detector
        self._last_gesture_debug_details: dict[str, Any] | None = None
        self._last_attention_debug_details: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraAdapter":
        """Build one adapter directly from ``TwinrConfig``."""

        return cls(
            config=AICameraAdapterConfig.from_config(config),
            face_anchor_detector=OpenCVFaceAnchorDetector.from_runtime_config(config),
        )

    def close(self) -> None:
        """Close any active Picamera2 and MediaPipe sessions."""

        timeout_s = self._lock_timeout_s()  # AUDIT-FIX(#4): Clamp invalid timeout config to a bounded safe value.
        if not self._lock.acquire(timeout=timeout_s):
            logger.warning("Timed out waiting to acquire AI camera adapter lock during close.")
            return
        try:
            self._reset_runtime_state_locked(close_pipeline=True, clear_pose=True, clear_motion=True)  # AUDIT-FIX(#1): Serialize teardown with observe to avoid mid-flight closure.
        finally:
            self._lock.release()
