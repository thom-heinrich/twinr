"""Capture bounded local-first IMX500 camera observations for Twinr.

This adapter owns the Raspberry Pi AI Camera runtime path. It probes camera
health, acquires one bounded inference frame, and returns explicit structured
signals instead of raw frame data so higher-level policy can stay decoupled
from Picamera2 and model-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any
import importlib
import logging
import math
import time

from twinr.agent.base_agent.config import TwinrConfig


logger = logging.getLogger(__name__)

_DEFAULT_DETECTION_NETWORK = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
_DEFAULT_POSE_NETWORK = "/usr/share/imx500-models/imx500_network_posenet.rpk"
_DEFAULT_POSE_BACKEND = "mediapipe"
_DEFAULT_MEDIAPIPE_POSE_MODEL = "state/mediapipe/models/pose_landmarker_full.task"
_DEFAULT_MEDIAPIPE_GESTURE_MODEL = "state/mediapipe/models/gesture_recognizer.task"
_DEFAULT_MAIN_SIZE = (640, 480)
_DEFAULT_FRAME_RATE = 10
_DEFAULT_LOCK_TIMEOUT_S = 5.0
_DEFAULT_STARTUP_WARMUP_S = 0.8
_DEFAULT_METADATA_WAIT_S = 3.0
_DEFAULT_PERSON_CONFIDENCE = 0.40
_DEFAULT_OBJECT_CONFIDENCE = 0.55
_DEFAULT_PERSON_NEAR_AREA = 0.20
_DEFAULT_PERSON_NEAR_HEIGHT = 0.55
_DEFAULT_OBJECT_NEAR_AREA = 0.08
_DEFAULT_ATTENTION_SCORE_THRESHOLD = 0.62
_DEFAULT_ENGAGED_SCORE_THRESHOLD = 0.45
_DEFAULT_POSE_CONFIDENCE = 0.30
_DEFAULT_POSE_REFRESH_S = 12.0
_DEFAULT_SEQUENCE_WINDOW_S = 1.6
_DEFAULT_SEQUENCE_MIN_FRAMES = 4
_DEFAULT_UNDEFINED_LABELS = frozenset({"-", "", "unknown"})
_PERSON_LABELS = frozenset({"person"})
_KEYPOINT_COUNT = 17
_NO_GESTURE = "none"
_MOTION_UNKNOWN_MAX_GAP_S = 12.0
_MOTION_MIN_DELTA_S = 0.15


class AICameraZone(StrEnum):
    """Describe one coarse horizontal zone."""

    UNKNOWN = "unknown"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AICameraBodyPose(StrEnum):
    """Describe one coarse body pose."""

    UNKNOWN = "unknown"
    UPRIGHT = "upright"
    SEATED = "seated"
    SLUMPED = "slumped"
    LYING_LOW = "lying_low"
    FLOOR = "floor"


class AICameraMotionState(StrEnum):
    """Describe one coarse motion class derived from recent person-box deltas."""

    UNKNOWN = "unknown"
    STILL = "still"
    WALKING = "walking"
    APPROACHING = "approaching"
    LEAVING = "leaving"


class AICameraGestureEvent(StrEnum):
    """Describe the bounded coarse-arm gesture vocabulary."""

    NONE = "none"
    WAVE = "wave"
    STOP = "stop"
    DISMISS = "dismiss"
    CONFIRM = "confirm"
    ARMS_CROSSED = "arms_crossed"
    TWO_HAND_DISMISS = "two_hand_dismiss"
    TIMEOUT_T = "timeout_t"
    UNKNOWN = "unknown"


class AICameraFineHandGesture(StrEnum):
    """Describe the bounded fine-hand gesture vocabulary."""

    NONE = "none"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    POINTING = "pointing"
    OPEN_PALM = "open_palm"
    OK_SIGN = "ok_sign"
    MIDDLE_FINGER = "middle_finger"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class AICameraAdapterConfig:
    """Store bounded runtime settings for the local IMX500 adapter."""

    detection_network_path: str = _DEFAULT_DETECTION_NETWORK
    pose_network_path: str = _DEFAULT_POSE_NETWORK
    pose_backend: str = _DEFAULT_POSE_BACKEND
    mediapipe_pose_model_path: str = _DEFAULT_MEDIAPIPE_POSE_MODEL
    mediapipe_gesture_model_path: str = _DEFAULT_MEDIAPIPE_GESTURE_MODEL
    mediapipe_custom_gesture_model_path: str | None = None
    mediapipe_num_hands: int = 2
    main_size: tuple[int, int] = _DEFAULT_MAIN_SIZE
    frame_rate: int = _DEFAULT_FRAME_RATE
    lock_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S
    startup_warmup_s: float = _DEFAULT_STARTUP_WARMUP_S
    metadata_wait_s: float = _DEFAULT_METADATA_WAIT_S
    person_confidence_threshold: float = _DEFAULT_PERSON_CONFIDENCE
    object_confidence_threshold: float = _DEFAULT_OBJECT_CONFIDENCE
    person_near_area_threshold: float = _DEFAULT_PERSON_NEAR_AREA
    person_near_height_threshold: float = _DEFAULT_PERSON_NEAR_HEIGHT
    object_near_area_threshold: float = _DEFAULT_OBJECT_NEAR_AREA
    attention_score_threshold: float = _DEFAULT_ATTENTION_SCORE_THRESHOLD
    engaged_score_threshold: float = _DEFAULT_ENGAGED_SCORE_THRESHOLD
    pose_confidence_threshold: float = _DEFAULT_POSE_CONFIDENCE
    pose_refresh_s: float = _DEFAULT_POSE_REFRESH_S
    sequence_window_s: float = _DEFAULT_SEQUENCE_WINDOW_S
    sequence_min_frames: int = _DEFAULT_SEQUENCE_MIN_FRAMES

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AICameraAdapterConfig":
        """Build one local AI-camera config from ``TwinrConfig``."""

        return cls(
            detection_network_path=str(
                getattr(config, "proactive_local_camera_detection_network_path", _DEFAULT_DETECTION_NETWORK)
                or _DEFAULT_DETECTION_NETWORK
            ),
            pose_network_path=str(getattr(config, "proactive_local_camera_pose_network_path", _DEFAULT_POSE_NETWORK) or _DEFAULT_POSE_NETWORK),
            pose_backend=str(getattr(config, "proactive_local_camera_pose_backend", _DEFAULT_POSE_BACKEND) or _DEFAULT_POSE_BACKEND).strip().lower(),
            mediapipe_pose_model_path=str(
                getattr(config, "proactive_local_camera_mediapipe_pose_model_path", _DEFAULT_MEDIAPIPE_POSE_MODEL)
                or _DEFAULT_MEDIAPIPE_POSE_MODEL
            ),
            mediapipe_gesture_model_path=str(
                getattr(config, "proactive_local_camera_mediapipe_gesture_model_path", _DEFAULT_MEDIAPIPE_GESTURE_MODEL)
                or _DEFAULT_MEDIAPIPE_GESTURE_MODEL
            ),
            mediapipe_custom_gesture_model_path=(
                str(getattr(config, "proactive_local_camera_mediapipe_custom_gesture_model_path", "") or "").strip()
                or None
            ),
            mediapipe_num_hands=_coerce_positive_int(
                getattr(config, "proactive_local_camera_mediapipe_num_hands", 2),
                default=2,
            ),
            main_size=(
                _coerce_positive_int(getattr(config, "camera_width", _DEFAULT_MAIN_SIZE[0]), default=_DEFAULT_MAIN_SIZE[0]),
                _coerce_positive_int(getattr(config, "camera_height", _DEFAULT_MAIN_SIZE[1]), default=_DEFAULT_MAIN_SIZE[1]),
            ),
            frame_rate=_coerce_positive_int(
                getattr(config, "proactive_local_camera_frame_rate", _DEFAULT_FRAME_RATE),
                default=_DEFAULT_FRAME_RATE,
            ),
            lock_timeout_s=_coerce_positive_float(
                getattr(config, "proactive_local_camera_lock_timeout_s", _DEFAULT_LOCK_TIMEOUT_S),
                default=_DEFAULT_LOCK_TIMEOUT_S,
            ),
            startup_warmup_s=_coerce_non_negative_float(
                getattr(config, "proactive_local_camera_startup_warmup_s", _DEFAULT_STARTUP_WARMUP_S),
                default=_DEFAULT_STARTUP_WARMUP_S,
            ),
            metadata_wait_s=_coerce_positive_float(
                getattr(config, "proactive_local_camera_metadata_wait_s", _DEFAULT_METADATA_WAIT_S),
                default=_DEFAULT_METADATA_WAIT_S,
            ),
            person_confidence_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_person_confidence_threshold", _DEFAULT_PERSON_CONFIDENCE),
                default=_DEFAULT_PERSON_CONFIDENCE,
            ),
            object_confidence_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_object_confidence_threshold", _DEFAULT_OBJECT_CONFIDENCE),
                default=_DEFAULT_OBJECT_CONFIDENCE,
            ),
            person_near_area_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_person_near_area_threshold", _DEFAULT_PERSON_NEAR_AREA),
                default=_DEFAULT_PERSON_NEAR_AREA,
            ),
            person_near_height_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_person_near_height_threshold", _DEFAULT_PERSON_NEAR_HEIGHT),
                default=_DEFAULT_PERSON_NEAR_HEIGHT,
            ),
            object_near_area_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_object_near_area_threshold", _DEFAULT_OBJECT_NEAR_AREA),
                default=_DEFAULT_OBJECT_NEAR_AREA,
            ),
            attention_score_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_attention_score_threshold", _DEFAULT_ATTENTION_SCORE_THRESHOLD),
                default=_DEFAULT_ATTENTION_SCORE_THRESHOLD,
            ),
            engaged_score_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_engaged_score_threshold", _DEFAULT_ENGAGED_SCORE_THRESHOLD),
                default=_DEFAULT_ENGAGED_SCORE_THRESHOLD,
            ),
            pose_confidence_threshold=_clamp_ratio(
                getattr(config, "proactive_local_camera_pose_confidence_threshold", _DEFAULT_POSE_CONFIDENCE),
                default=_DEFAULT_POSE_CONFIDENCE,
            ),
            pose_refresh_s=_coerce_non_negative_float(
                getattr(config, "proactive_local_camera_pose_refresh_s", _DEFAULT_POSE_REFRESH_S),
                default=_DEFAULT_POSE_REFRESH_S,
            ),
            sequence_window_s=_coerce_positive_float(
                getattr(config, "proactive_local_camera_sequence_window_s", _DEFAULT_SEQUENCE_WINDOW_S),
                default=_DEFAULT_SEQUENCE_WINDOW_S,
            ),
            sequence_min_frames=_coerce_positive_int(
                getattr(config, "proactive_local_camera_sequence_min_frames", _DEFAULT_SEQUENCE_MIN_FRAMES),
                default=_DEFAULT_SEQUENCE_MIN_FRAMES,
            ),
        )


@dataclass(frozen=True, slots=True)
class AICameraBox:
    """Describe one normalized bounding box."""

    top: float
    left: float
    bottom: float
    right: float

    def __post_init__(self) -> None:
        """Clamp and order the normalized edges."""

        top = _clamp_ratio(self.top, default=0.0)
        left = _clamp_ratio(self.left, default=0.0)
        bottom = _clamp_ratio(self.bottom, default=top)
        right = _clamp_ratio(self.right, default=left)
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
    def width(self) -> float:
        """Return the normalized width."""

        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        """Return the normalized height."""

        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        """Return the normalized area."""

        return self.width * self.height


@dataclass(frozen=True, slots=True)
class AICameraObjectDetection:
    """Describe one local object detection before surface stabilization."""

    label: str
    confidence: float
    zone: AICameraZone
    box: AICameraBox | None = None

    def __post_init__(self) -> None:
        """Normalize detection metadata into inspectable values."""

        label = _normalize_label(self.label)
        object.__setattr__(self, "label", label if label else "unknown")
        object.__setattr__(self, "confidence", _clamp_ratio(self.confidence, default=0.0))
        object.__setattr__(self, "zone", _coerce_zone(self.zone))
        object.__setattr__(self, "box", self.box)


@dataclass(frozen=True, slots=True)
class AICameraObservation:
    """Describe one bounded IMX500 observation tick."""

    observed_at: float
    camera_online: bool
    camera_ready: bool
    camera_ai_ready: bool
    camera_error: str | None = None
    last_camera_frame_at: float | None = None
    last_camera_health_change_at: float | None = None
    person_count: int = 0
    primary_person_box: AICameraBox | None = None
    primary_person_zone: AICameraZone = AICameraZone.UNKNOWN
    looking_toward_device: bool | None = None
    person_near_device: bool | None = None
    engaged_with_device: bool | None = None
    visual_attention_score: float | None = None
    body_pose: AICameraBodyPose = AICameraBodyPose.UNKNOWN
    pose_confidence: float | None = None
    motion_state: AICameraMotionState = AICameraMotionState.UNKNOWN
    motion_confidence: float | None = None
    hand_or_object_near_camera: bool = False
    showing_intent_likely: bool | None = None
    gesture_event: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_confidence: float | None = None
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    objects: tuple[AICameraObjectDetection, ...] = ()
    model: str = "local-imx500"

    @property
    def primary_person_center_x(self) -> float | None:
        """Return the primary-person horizontal center when available."""

        if self.primary_person_box is None:
            return None
        return self.primary_person_box.center_x

    @property
    def primary_person_center_y(self) -> float | None:
        """Return the primary-person vertical center when available."""

        if self.primary_person_box is None:
            return None
        return self.primary_person_box.center_y


@dataclass(slots=True)
class _NetworkSession:
    """Store one active Picamera2 session and its loaded IMX500 network."""

    network_path: str
    task_name: str
    picam2: Any
    imx500: Any
    input_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _DetectionResult:
    """Describe one parsed detection frame."""

    person_count: int
    primary_person_box: AICameraBox | None
    primary_person_zone: AICameraZone
    person_near_device: bool | None
    hand_or_object_near_camera: bool
    objects: tuple[AICameraObjectDetection, ...]


@dataclass(frozen=True, slots=True)
class _PoseResult:
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


@dataclass(frozen=True, slots=True)
class _PoseCandidateMatch:
    """Describe how one HigherHRNet candidate aligns with the primary person."""

    candidate_index: int
    raw_keypoints: list[float]
    raw_score: float
    normalized_score: float
    box: AICameraBox
    overlap: float
    center_similarity: float
    size_similarity: float
    selection_score: float


class LocalAICameraAdapter:
    """Provide one bounded local-first IMX500 observation surface."""

    def __init__(
        self,
        *,
        config: AICameraAdapterConfig | None = None,
        clock: Any = time.time,
        sleep_fn: Any = time.sleep,
    ) -> None:
        """Initialize one bounded adapter with lazy Picamera2 imports."""

        self.config = config or AICameraAdapterConfig()
        self._clock = clock
        self._sleep = sleep_fn
        self._lock = Lock()
        self._session: _NetworkSession | None = None
        self._last_frame_at: float | None = None
        self._last_health_change_at: float | None = None
        self._last_health_signature: tuple[bool, bool, bool, str | None] | None = None
        self._last_pose_at: float | None = None
        self._last_pose_result: _PoseResult | None = None
        self._mediapipe_pipeline: Any | None = None
        self._last_motion_box: AICameraBox | None = None
        self._last_motion_person_count = 0
        self._last_motion_at: float | None = None
        self._last_motion_state = AICameraMotionState.UNKNOWN
        self._last_motion_confidence: float | None = None

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "LocalAICameraAdapter":
        """Build one adapter directly from ``TwinrConfig``."""

        return cls(config=AICameraAdapterConfig.from_config(config))

    def close(self) -> None:
        """Close any active Picamera2 session."""

        self._close_session()
        if self._mediapipe_pipeline is not None:
            try:
                self._mediapipe_pipeline.close()
            except Exception:  # pragma: no cover - depends on MediaPipe runtime state.
                logger.debug("Ignoring MediaPipe close failure during AI camera cleanup.", exc_info=True)
            self._mediapipe_pipeline = None

    def observe(self) -> AICameraObservation:
        """Capture one local IMX500 observation or one explicit health failure."""

        observed_at = self._now()
        if not self._lock.acquire(timeout=self.config.lock_timeout_s):
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error="camera_lock_timeout",
            )
        try:
            try:
                runtime = self._load_detection_runtime()
            except Exception as exc:  # pragma: no cover - depends on local environment.
                code = self._classify_error(exc)
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=code,
                )
            online_error = self._probe_online(runtime)
            if online_error is not None:
                return self._health_only_observation(
                    observed_at=observed_at,
                    online=False,
                    ready=False,
                    ai_ready=False,
                    error=online_error,
                )

            detection = self._capture_detection(runtime, observed_at=observed_at)
            pose_result, pose_error = self._resolve_pose(runtime, observed_at=observed_at, detection=detection)
            observation = self._compose_observation(
                observed_at=observed_at,
                detection=detection,
                pose=pose_result,
                pose_error=pose_error,
            )
            self._last_frame_at = observed_at
            return self._with_health(
                observation,
                online=True,
                ready=True,
                ai_ready=(pose_error is None),
                error=pose_error,
                frame_at=observed_at,
            )
        except Exception as exc:  # pragma: no cover - hardware and library behavior are environment-dependent.
            code = self._classify_error(exc)
            logger.warning("Local AI camera observation failed with %s: %s", code, exc)
            self._close_session()
            return self._health_only_observation(
                observed_at=observed_at,
                online=True,
                ready=False,
                ai_ready=False,
                error=code,
            )
        finally:
            self._lock.release()

    def _resolve_pose(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        detection: _DetectionResult,
    ) -> tuple[_PoseResult | None, str | None]:
        """Return one fresh or cached pose result for the current detection frame."""

        if detection.person_count <= 0:
            if self._mediapipe_pipeline is not None:
                self._mediapipe_pipeline.reset_temporal_state()
            self._last_pose_result = None
            self._last_pose_at = None
            return None, None

        if self.config.pose_backend == "mediapipe":
            return self._resolve_mediapipe_pose(
                runtime,
                observed_at=observed_at,
                detection=detection,
            )
        if not self.config.pose_network_path:
            self._last_pose_result = None
            self._last_pose_at = None
            return None, None

        if (
            self._last_pose_result is not None
            and self._last_pose_at is not None
            and self.config.pose_refresh_s > 0.0
            and (observed_at - self._last_pose_at) <= self.config.pose_refresh_s
        ):
            return self._last_pose_result, None

        try:
            pose_postprocess = self._load_pose_postprocess()
        except ImportError:
            return None, "pose_dependency_missing"

        try:
            pose = self._capture_pose(
                runtime,
                pose_postprocess=pose_postprocess,
                observed_at=observed_at,
                primary_person_box=detection.primary_person_box,
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent path.
            logger.warning("Local AI camera pose decode failed: %s", exc)
            return None, self._classify_error(exc)

        self._last_pose_result = pose
        self._last_pose_at = observed_at
        return pose, None

    def _resolve_mediapipe_pose(
        self,
        runtime: dict[str, Any],
        *,
        observed_at: float,
        detection: _DetectionResult,
    ) -> tuple[_PoseResult | None, str | None]:
        """Run the Pi-side MediaPipe pose and gesture path on the RGB preview frame."""

        if detection.primary_person_box is None:
            return None, None
        try:
            frame_rgb = self._capture_rgb_frame(runtime, observed_at=observed_at)
            pipeline = self._ensure_mediapipe_pipeline()
            result = pipeline.analyze(
                frame_rgb=frame_rgb,
                observed_at=observed_at,
                primary_person_box=detection.primary_person_box,
            )
        except Exception as exc:  # pragma: no cover - depends on Pi runtime and model assets.
            logger.warning("Local AI camera MediaPipe inference failed: %s", exc)
            return None, self._classify_error(exc)
        pose = _PoseResult(
            body_pose=result.body_pose,
            pose_confidence=result.pose_confidence,
            looking_toward_device=result.looking_toward_device,
            visual_attention_score=result.visual_attention_score,
            hand_near_camera=result.hand_near_camera,
            showing_intent_likely=result.showing_intent_likely,
            gesture_event=result.gesture_event,
            gesture_confidence=result.gesture_confidence,
            fine_hand_gesture=result.fine_hand_gesture,
            fine_hand_gesture_confidence=result.fine_hand_gesture_confidence,
        )
        self._last_pose_result = pose
        self._last_pose_at = observed_at
        return pose, None

    def _compose_observation(
        self,
        *,
        observed_at: float,
        detection: _DetectionResult,
        pose: _PoseResult | None,
        pose_error: str | None,
    ) -> AICameraObservation:
        """Merge detection and pose signals into one bounded observation."""

        primary_person_box = detection.primary_person_box
        primary_person_zone = detection.primary_person_zone
        person_near_device = detection.person_near_device

        visual_attention_score = None
        looking_toward_device = None
        engaged_with_device = None
        body_pose = AICameraBodyPose.UNKNOWN
        pose_confidence = None
        motion_state = AICameraMotionState.UNKNOWN
        motion_confidence = None
        gesture_event = AICameraGestureEvent.UNKNOWN if pose_error is not None else AICameraGestureEvent.NONE
        gesture_confidence = None
        fine_hand_gesture = AICameraFineHandGesture.UNKNOWN if pose_error is not None else AICameraFineHandGesture.NONE
        fine_hand_gesture_confidence = None
        showing_intent_likely = None
        hand_or_object_near_camera = detection.hand_or_object_near_camera

        if primary_person_box is not None:
            base_center_score = 1.0 - min(1.0, abs(primary_person_box.center_x - 0.5) / 0.5)
            visual_attention_score = round(base_center_score * 0.35, 3)
            looking_toward_device = visual_attention_score >= self.config.attention_score_threshold
            engaged_with_device = (
                person_near_device is True
                and visual_attention_score >= self.config.engaged_score_threshold
            )
            showing_intent_likely = hand_or_object_near_camera and (
                looking_toward_device or person_near_device is True
            )

        if pose is not None:
            visual_attention_score = pose.visual_attention_score
            looking_toward_device = pose.looking_toward_device
            engaged_with_device = (
                detection.person_count > 0
                and (person_near_device is True or pose.looking_toward_device is True)
                and (pose.visual_attention_score or 0.0) >= self.config.engaged_score_threshold
            )
            body_pose = pose.body_pose
            pose_confidence = pose.pose_confidence
            gesture_event = pose.gesture_event
            gesture_confidence = pose.gesture_confidence
            fine_hand_gesture = pose.fine_hand_gesture
            fine_hand_gesture_confidence = pose.fine_hand_gesture_confidence
            hand_or_object_near_camera = hand_or_object_near_camera or pose.hand_near_camera
            showing_intent_likely = pose.showing_intent_likely

        motion_state, motion_confidence = self._resolve_motion(
            observed_at=observed_at,
            person_count=detection.person_count,
            primary_person_box=primary_person_box,
        )

        return AICameraObservation(
            observed_at=observed_at,
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            camera_error=pose_error,
            person_count=detection.person_count,
            primary_person_box=primary_person_box,
            primary_person_zone=primary_person_zone,
            looking_toward_device=looking_toward_device,
            person_near_device=person_near_device,
            engaged_with_device=engaged_with_device,
            visual_attention_score=visual_attention_score,
            body_pose=body_pose,
            pose_confidence=pose_confidence,
            motion_state=motion_state,
            motion_confidence=motion_confidence,
            hand_or_object_near_camera=hand_or_object_near_camera,
            showing_intent_likely=showing_intent_likely,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_gesture_confidence,
            objects=detection.objects,
            model=(
                "local-imx500+mediapipe"
                if self.config.pose_backend == "mediapipe"
                else "local-imx500"
            ),
        )

    def _capture_detection(self, runtime: dict[str, Any], *, observed_at: float) -> _DetectionResult:
        """Capture one detection frame from the SSD network."""

        session = self._ensure_session(
            runtime,
            network_path=self.config.detection_network_path,
            task_name="detection",
        )
        metadata = self._capture_metadata(session, observed_at=observed_at)
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
        if not outputs or len(outputs) < 4:
            raise RuntimeError("detection_outputs_missing")

        labels = tuple(getattr(session.imx500.network_intrinsics, "labels", ()) or ())
        count = min(100, max(0, int(round(float(outputs[3][0][0])))))
        person_boxes: list[tuple[float, AICameraBox]] = []
        object_detections: list[AICameraObjectDetection] = []

        for index in range(count):
            score = _coerce_float(outputs[1][0][index], default=0.0)
            class_index = max(0, int(round(_coerce_float(outputs[2][0][index], default=0.0))))
            label = labels[class_index] if class_index < len(labels) else f"class_{class_index}"
            normalized_label = _normalize_label(label)
            if normalized_label in _DEFAULT_UNDEFINED_LABELS:
                continue
            box = _box_from_detection(outputs[0][0][index])
            if normalized_label in _PERSON_LABELS:
                if score >= self.config.person_confidence_threshold:
                    person_boxes.append((score, box))
                continue
            if score < self.config.object_confidence_threshold:
                continue
            object_detections.append(
                AICameraObjectDetection(
                    label=normalized_label,
                    confidence=score,
                    zone=_zone_from_center(box.center_x),
                    box=box,
                )
            )

        person_boxes.sort(key=lambda item: item[0], reverse=True)
        primary_person_box = person_boxes[0][1] if person_boxes else None
        person_count = len(person_boxes)
        person_near_device = None
        primary_zone = AICameraZone.UNKNOWN
        if primary_person_box is not None:
            primary_zone = _zone_from_center(primary_person_box.center_x)
            person_near_device = (
                primary_person_box.area >= self.config.person_near_area_threshold
                or primary_person_box.height >= self.config.person_near_height_threshold
            )

        hand_or_object_near_camera = any(
            detection.box is not None and detection.box.area >= self.config.object_near_area_threshold
            for detection in object_detections
        )
        return _DetectionResult(
            person_count=person_count,
            primary_person_box=primary_person_box,
            primary_person_zone=primary_zone,
            person_near_device=person_near_device,
            hand_or_object_near_camera=hand_or_object_near_camera,
            objects=tuple(object_detections),
        )

    def _capture_pose(
        self,
        runtime: dict[str, Any],
        *,
        pose_postprocess: Any,
        observed_at: float,
        primary_person_box: AICameraBox | None,
    ) -> _PoseResult:
        """Capture one pose frame and decode one coarse pose sample."""

        session = self._ensure_session(
            runtime,
            network_path=self.config.pose_network_path,
            task_name="pose",
        )
        metadata = self._capture_metadata(session, observed_at=observed_at)
        outputs = session.imx500.get_outputs(metadata, add_batch=True)
        if not outputs or len(outputs) < 3:
            raise RuntimeError("pose_outputs_missing")
        normalized_outputs = list(outputs)
        if (
            len(normalized_outputs) >= 2
            and normalized_outputs[0].shape[-1] == 17
            and normalized_outputs[1].shape[-1] == 34
        ):
            # The Pi firmware currently returns the 17-channel heatmap tensor
            # before the 34-channel heatmap+tag tensor, while the official
            # HigherHRNet postprocess expects the opposite order.
            normalized_outputs[0], normalized_outputs[1] = normalized_outputs[1], normalized_outputs[0]

        input_width, input_height = session.input_size
        keypoints, scores, bboxes = pose_postprocess(
            normalized_outputs,
            (self.config.main_size[1], self.config.main_size[0]),
            (0, 0),
            (0, 0),
            False,
            input_image_size=(input_height, input_width),
            output_shape=(normalized_outputs[0].shape[1], normalized_outputs[0].shape[2]),
        )
        if not keypoints or not scores or not bboxes:
            raise RuntimeError("pose_people_missing")

        selected_keypoints, selected_score, selected_box = self._select_primary_pose(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=primary_person_box,
        )
        parsed_keypoints = _parse_keypoints(
            selected_keypoints,
            frame_width=self.config.main_size[0],
            frame_height=self.config.main_size[1],
        )
        pose_confidence = _support_pose_confidence(
            selected_score,
            parsed_keypoints,
            fallback_box=selected_box,
        )
        if pose_confidence < self.config.pose_confidence_threshold:
            raise RuntimeError("pose_confidence_low")
        attention_score = _attention_score(parsed_keypoints, fallback_box=selected_box)
        looking_toward_device = attention_score >= self.config.attention_score_threshold
        hand_near_camera = _hand_near_camera(parsed_keypoints, fallback_box=selected_box)
        gesture_event, gesture_confidence = _classify_gesture(
            parsed_keypoints,
            attention_score=attention_score,
            fallback_box=selected_box,
        )
        return _PoseResult(
            body_pose=_classify_body_pose(parsed_keypoints, fallback_box=selected_box),
            pose_confidence=pose_confidence,
            looking_toward_device=looking_toward_device,
            visual_attention_score=attention_score,
            hand_near_camera=hand_near_camera,
            showing_intent_likely=hand_near_camera and looking_toward_device,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
        )

    def _capture_rgb_frame(self, runtime: dict[str, Any], *, observed_at: float) -> Any:
        """Capture one RGB preview frame from the live detection session."""

        session = self._ensure_session(
            runtime,
            network_path=self.config.detection_network_path,
            task_name="detection",
        )
        try:
            frame = session.picam2.capture_array("main")
        except TypeError:
            frame = session.picam2.capture_array()
        if frame is None:
            raise RuntimeError("rgb_frame_missing")
        return frame

    def _ensure_mediapipe_pipeline(self) -> Any:
        """Reuse or create the Pi-side MediaPipe inference pipeline lazily."""

        if self._mediapipe_pipeline is not None:
            return self._mediapipe_pipeline
        from twinr.hardware.mediapipe_vision import MediaPipeVisionConfig, MediaPipeVisionPipeline

        self._mediapipe_pipeline = MediaPipeVisionPipeline(
            config=MediaPipeVisionConfig.from_ai_camera_config(self.config),
        )
        return self._mediapipe_pipeline

    def _select_primary_pose(
        self,
        *,
        keypoints: list[list[float]],
        scores: list[float],
        bboxes: list[list[float]],
        primary_person_box: AICameraBox | None,
    ) -> tuple[list[float], float, AICameraBox]:
        """Return the pose sample that best matches the primary person."""

        candidates = _rank_pose_candidates(
            keypoints=keypoints,
            scores=scores,
            bboxes=bboxes,
            primary_person_box=primary_person_box,
            frame_width=self.config.main_size[0],
            frame_height=self.config.main_size[1],
        )
        if not candidates:
            raise RuntimeError("pose_people_missing")
        selected = candidates[0]
        return selected.raw_keypoints, selected.raw_score, selected.box

    def _capture_metadata(self, session: _NetworkSession, *, observed_at: float) -> dict[str, Any]:
        """Wait for the next metadata payload that contains CNN tensors."""

        deadline = time.monotonic() + self.config.metadata_wait_s
        last_metadata: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            metadata = session.picam2.capture_metadata()
            if isinstance(metadata, dict):
                last_metadata = metadata
                if metadata.get("CnnOutputTensor"):
                    return metadata
            self._sleep(0.05)
        logger.warning(
            "Timed out waiting for IMX500 metadata on %s after %.2fs.",
            session.task_name,
            self.config.metadata_wait_s,
        )
        if last_metadata is not None:
            return last_metadata
        raise RuntimeError("metadata_timeout")

    def _ensure_session(
        self,
        runtime: dict[str, Any],
        *,
        network_path: str,
        task_name: str,
    ) -> _NetworkSession:
        """Reuse or create one Picamera2 session for the requested network."""

        if self._session is not None and self._session.network_path == network_path:
            return self._session

        self._close_session()
        if not Path(network_path).exists():
            raise FileNotFoundError(f"model_missing:{network_path}")

        IMX500 = runtime["IMX500"]
        Picamera2 = runtime["Picamera2"]
        imx500 = IMX500(network_path)
        picam2 = Picamera2(imx500.camera_num)
        configuration = picam2.create_preview_configuration(
            main={"size": self.config.main_size, "format": "RGB888"},
            controls={"FrameRate": self.config.frame_rate},
        )
        picam2.configure(configuration)
        picam2.start()
        if self.config.startup_warmup_s > 0:
            self._sleep(self.config.startup_warmup_s)
        self._session = _NetworkSession(
            network_path=network_path,
            task_name=task_name,
            picam2=picam2,
            imx500=imx500,
            input_size=tuple(imx500.get_input_size()),
        )
        return self._session

    def _close_session(self) -> None:
        """Stop and discard the current Picamera2 session."""

        session = self._session
        self._session = None
        if session is None:
            return
        try:
            session.picam2.stop()
        except Exception:  # pragma: no cover - depends on camera state.
            logger.debug("Ignoring Picamera2 stop failure during AI camera cleanup.", exc_info=True)
        if hasattr(session.picam2, "close"):
            try:
                session.picam2.close()
            except Exception:  # pragma: no cover - depends on camera state.
                logger.debug("Ignoring Picamera2 close failure during AI camera cleanup.", exc_info=True)

    def _probe_online(self, runtime: dict[str, Any]) -> str | None:
        """Return ``None`` when an IMX500 camera is enumerated, else one error code."""

        Picamera2 = runtime["Picamera2"]
        try:
            camera_info = Picamera2.global_camera_info()
        except Exception as exc:  # pragma: no cover - depends on libcamera runtime.
            logger.warning("IMX500 enumeration failed: %s", exc)
            return "camera_probe_failed"
        for item in camera_info or ():
            if str(getattr(item, "get", lambda *_: "")("Model", "")).lower() == "imx500":
                return None
            if str(item.get("Model", "")).lower() == "imx500":
                return None
        return "imx500_not_enumerated"

    def _load_detection_runtime(self) -> dict[str, Any]:
        """Import the minimum Picamera2 runtime needed for SSD detection."""

        try:
            picamera2_module = importlib.import_module("picamera2")
            imx500_module = importlib.import_module("picamera2.devices.imx500")
        except Exception as exc:  # pragma: no cover - depends on local environment.
            raise RuntimeError("picamera2_unavailable") from exc
        return {
            "Picamera2": getattr(picamera2_module, "Picamera2"),
            "IMX500": getattr(imx500_module, "IMX500"),
        }

    def _load_pose_postprocess(self) -> Any:
        """Import the HigherHRNet postprocess helper when available."""

        module = importlib.import_module("picamera2.devices.imx500.postprocess_highernet")
        return getattr(module, "postprocess_higherhrnet")

    def _health_only_observation(
        self,
        *,
        observed_at: float,
        online: bool,
        ready: bool,
        ai_ready: bool,
        error: str | None,
    ) -> AICameraObservation:
        """Return one explicit health-only observation."""

        observation = AICameraObservation(
            observed_at=observed_at,
            camera_online=online,
            camera_ready=ready,
            camera_ai_ready=ai_ready,
            camera_error=error,
        )
        return self._with_health(observation, online=online, ready=ready, ai_ready=ai_ready, error=error, frame_at=None)

    def _with_health(
        self,
        observation: AICameraObservation,
        *,
        online: bool,
        ready: bool,
        ai_ready: bool,
        error: str | None,
        frame_at: float | None,
    ) -> AICameraObservation:
        """Attach the latest health timestamps to one observation."""

        health_signature = (online, ready, ai_ready, error)
        if health_signature != self._last_health_signature:
            self._last_health_signature = health_signature
            self._last_health_change_at = observation.observed_at
        return AICameraObservation(
            observed_at=observation.observed_at,
            camera_online=online,
            camera_ready=ready,
            camera_ai_ready=ai_ready,
            camera_error=error,
            last_camera_frame_at=(frame_at if frame_at is not None else self._last_frame_at),
            last_camera_health_change_at=self._last_health_change_at,
            person_count=observation.person_count,
            primary_person_box=observation.primary_person_box,
            primary_person_zone=observation.primary_person_zone,
            looking_toward_device=observation.looking_toward_device,
            person_near_device=observation.person_near_device,
            engaged_with_device=observation.engaged_with_device,
            visual_attention_score=observation.visual_attention_score,
            body_pose=observation.body_pose,
            pose_confidence=observation.pose_confidence,
            motion_state=observation.motion_state,
            motion_confidence=observation.motion_confidence,
            hand_or_object_near_camera=observation.hand_or_object_near_camera,
            showing_intent_likely=observation.showing_intent_likely,
            gesture_event=observation.gesture_event,
            gesture_confidence=observation.gesture_confidence,
            fine_hand_gesture=observation.fine_hand_gesture,
            fine_hand_gesture_confidence=observation.fine_hand_gesture_confidence,
            objects=observation.objects,
            model=observation.model,
        )

    def _resolve_motion(
        self,
        *,
        observed_at: float,
        person_count: int,
        primary_person_box: AICameraBox | None,
    ) -> tuple[AICameraMotionState, float | None]:
        """Derive one coarse motion state from recent primary-person box deltas."""

        motion_state, motion_confidence = _infer_motion_state(
            previous_box=self._last_motion_box,
            current_box=primary_person_box,
            previous_observed_at=self._last_motion_at,
            current_observed_at=observed_at,
            previous_person_count=self._last_motion_person_count,
            current_person_count=person_count,
        )
        self._last_motion_box = primary_person_box
        self._last_motion_person_count = max(0, int(person_count))
        self._last_motion_at = observed_at
        self._last_motion_state = motion_state
        self._last_motion_confidence = motion_confidence
        return motion_state, motion_confidence

    def _classify_error(self, exc: Exception) -> str:
        """Return one stable operator-facing error code for the exception."""

        message = str(exc).strip().lower()
        if "camera __init__ sequence did not complete" in message or "device or resource busy" in message:
            return "camera_busy"
        if "requested camera dev-node not found" in message:
            return "imx500_not_enumerated"
        if "picamera2_unavailable" in message:
            return "picamera2_unavailable"
        if "metadata_timeout" in message:
            return "metadata_timeout"
        if "mediapipe_custom_gesture_model_missing" in message:
            return "mediapipe_custom_gesture_model_missing"
        if "mediapipe_pose_model_missing" in message:
            return "mediapipe_pose_model_missing"
        if "mediapipe_gesture_model_missing" in message:
            return "mediapipe_gesture_model_missing"
        if "mediapipe_unavailable" in message:
            return "mediapipe_unavailable"
        if "model_missing" in message:
            return "model_missing"
        if "pose_dependency_missing" in message:
            return "pose_dependency_missing"
        if "rgb_frame_missing" in message:
            return "rgb_frame_missing"
        if "pose_outputs_missing" in message or "pose_people_missing" in message or "operands could not be broadcast together" in message:
            return "pose_decode_failed"
        if "pose_confidence_low" in message:
            return "pose_confidence_low"
        if isinstance(exc, FileNotFoundError):
            return "model_missing"
        return exc.__class__.__name__.lower()

    def _now(self) -> float:
        """Return one bounded wall-clock timestamp."""

        try:
            value = float(self._clock())
        except Exception:
            return time.time()
        if not math.isfinite(value) or value < 0.0:
            return time.time()
        return value


def _normalize_label(value: object) -> str:
    """Normalize one object label to an inspectable token."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text[:64]


def _coerce_zone(value: object) -> AICameraZone:
    """Coerce one zone-like payload to ``AICameraZone``."""

    if isinstance(value, AICameraZone):
        return value
    try:
        return AICameraZone(str(value or "").strip().lower())
    except ValueError:
        return AICameraZone.UNKNOWN


def _coerce_float(value: object, *, default: float) -> float:
    """Coerce one value to a finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce one value to a positive integer."""

    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return number


def _coerce_positive_float(value: object, *, default: float) -> float:
    """Coerce one value to a positive finite float."""

    number = _coerce_float(value, default=default)
    if number <= 0.0:
        return default
    return number


def _coerce_non_negative_float(value: object, *, default: float) -> float:
    """Coerce one value to a non-negative finite float."""

    number = _coerce_float(value, default=default)
    if number < 0.0:
        return default
    return number


def _clamp_ratio(value: object, *, default: float) -> float:
    """Clamp one numeric value into ``[0.0, 1.0]``."""

    number = _coerce_float(value, default=default)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _zone_from_center(center_x: float) -> AICameraZone:
    """Map one normalized ``x`` center to a coarse zone."""

    if center_x < (1.0 / 3.0):
        return AICameraZone.LEFT
    if center_x > (2.0 / 3.0):
        return AICameraZone.RIGHT
    return AICameraZone.CENTER


def _box_from_detection(value: Any) -> AICameraBox:
    """Build one normalized box from the SSD ``ymin,xmin,ymax,xmax`` tensor."""

    top, left, bottom, right = (float(item) for item in list(value)[:4])
    return AICameraBox(top=top, left=left, bottom=bottom, right=right)


def _box_from_pixel_bbox(value: list[float], *, frame_width: int, frame_height: int) -> AICameraBox:
    """Build one normalized box from one pixel-space ``ymin,xmin,ymax,xmax`` list."""

    top, left, bottom, right = (float(item) for item in list(value)[:4])
    return AICameraBox(
        top=(top / max(1, frame_height)),
        left=(left / max(1, frame_width)),
        bottom=(bottom / max(1, frame_height)),
        right=(right / max(1, frame_width)),
    )


def _iou(left: AICameraBox, right: AICameraBox) -> float:
    """Return the normalized IoU of two boxes."""

    intersection = _intersection_area(left, right)
    if intersection <= 0.0:
        return 0.0
    union = left.area + right.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _intersection_area(left: AICameraBox, right: AICameraBox) -> float:
    """Return the normalized intersection area of two boxes."""

    top = max(left.top, right.top)
    left_edge = max(left.left, right.left)
    bottom = min(left.bottom, right.bottom)
    right_edge = min(left.right, right.right)
    if bottom <= top or right_edge <= left_edge:
        return 0.0
    return (bottom - top) * (right_edge - left_edge)


def _box_center_similarity(left: AICameraBox, right: AICameraBox) -> float:
    """Return how close two box centers are on a normalized frame."""

    distance = math.hypot(left.center_x - right.center_x, left.center_y - right.center_y)
    return round(max(0.0, 1.0 - min(1.0, distance / math.sqrt(2.0))), 3)


def _box_size_similarity(left: AICameraBox, right: AICameraBox) -> float:
    """Return one conservative width-height similarity score for two boxes."""

    if left.width <= 0.0 or right.width <= 0.0 or left.height <= 0.0 or right.height <= 0.0:
        return 0.0
    width_similarity = min(left.width, right.width) / max(left.width, right.width)
    height_similarity = min(left.height, right.height) / max(left.height, right.height)
    return round(width_similarity * height_similarity, 3)


def _score_pose_candidate(
    *,
    candidate_index: int,
    raw_keypoints: list[float],
    raw_score: float,
    box: AICameraBox,
    primary_person_box: AICameraBox | None,
) -> _PoseCandidateMatch:
    """Score one pose candidate against the detection-space primary person."""

    normalized_score = _clamp_ratio(raw_score, default=0.0)
    overlap = 0.0
    center_similarity = 0.0
    size_similarity = 0.0
    selection_score = normalized_score
    if primary_person_box is not None:
        overlap = round(_iou(box, primary_person_box), 3)
        center_similarity = _box_center_similarity(box, primary_person_box)
        size_similarity = _box_size_similarity(box, primary_person_box)
        # The detection box is the authoritative anchor for "which person";
        # HigherHRNet raw scores can exceed 1.0 on the Pi and must not dominate.
        selection_score = round(
            0.70 * overlap + 0.20 * center_similarity + 0.08 * size_similarity + 0.02 * normalized_score,
            3,
        )
    return _PoseCandidateMatch(
        candidate_index=candidate_index,
        raw_keypoints=raw_keypoints,
        raw_score=raw_score,
        normalized_score=normalized_score,
        box=box,
        overlap=overlap,
        center_similarity=center_similarity,
        size_similarity=size_similarity,
        selection_score=selection_score,
    )


def _rank_pose_candidates(
    *,
    keypoints: list[list[float]],
    scores: list[float],
    bboxes: list[list[float]],
    primary_person_box: AICameraBox | None,
    frame_width: int,
    frame_height: int,
) -> list[_PoseCandidateMatch]:
    """Return pose candidates ranked by spatial alignment to the primary person."""

    candidates: list[_PoseCandidateMatch] = []
    for candidate_index, (raw_keypoints, raw_score, raw_bbox) in enumerate(zip(keypoints, scores, bboxes, strict=False)):
        score = _coerce_float(raw_score, default=0.0)
        box = _box_from_pixel_bbox(
            raw_bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        candidates.append(
            _score_pose_candidate(
                candidate_index=candidate_index,
                raw_keypoints=raw_keypoints,
                raw_score=score,
                box=box,
                primary_person_box=primary_person_box,
            )
        )
    candidates.sort(
        key=lambda item: (
            item.selection_score,
            item.overlap,
            item.center_similarity,
            item.size_similarity,
            item.normalized_score,
            item.raw_score,
        ),
        reverse=True,
    )
    return candidates


def _parse_keypoints(raw: list[float], *, frame_width: int, frame_height: int) -> dict[int, tuple[float, float, float]]:
    """Convert one flat keypoint list into normalized ``index -> (x, y, score)`` data."""

    parsed: dict[int, tuple[float, float, float]] = {}
    expected = min(len(raw) // 3, _KEYPOINT_COUNT)
    for index in range(expected):
        x = _clamp_ratio(raw[index * 3 + 0] / max(1, frame_width), default=0.0)
        y = _clamp_ratio(raw[index * 3 + 1] / max(1, frame_height), default=0.0)
        score = _clamp_ratio(raw[index * 3 + 2], default=0.0)
        parsed[index] = (x, y, score)
    return parsed


def _attention_score(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> float:
    """Return one conservative attention score from pose and coarse centering."""

    nose = keypoints.get(0)
    left_eye = keypoints.get(1)
    right_eye = keypoints.get(2)
    left_shoulder = keypoints.get(5)
    right_shoulder = keypoints.get(6)

    center_alignment = 1.0 - min(1.0, abs(fallback_box.center_x - 0.5) / 0.5)
    shoulder_alignment = 0.0
    head_alignment = 0.0

    if left_shoulder and right_shoulder and min(left_shoulder[2], right_shoulder[2]) >= 0.20:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_span = abs(right_shoulder[0] - left_shoulder[0])
        if nose and nose[2] >= 0.20 and shoulder_span > 0.02:
            head_alignment = max(0.0, 1.0 - abs(nose[0] - shoulder_center_x) / shoulder_span)
        shoulder_alignment = max(0.0, 1.0 - abs(shoulder_center_x - 0.5) / 0.5)
    elif nose and nose[2] >= 0.20 and left_eye and right_eye and min(left_eye[2], right_eye[2]) >= 0.20:
        eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_span = abs(right_eye[0] - left_eye[0])
        if eye_span > 0.01:
            head_alignment = max(0.0, 1.0 - abs(nose[0] - eye_center_x) / eye_span)

    score = 0.45 * head_alignment + 0.25 * shoulder_alignment + 0.30 * center_alignment
    return round(_clamp_ratio(score, default=0.0), 3)


def _hand_near_camera(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> bool:
    """Return whether one wrist is plausibly near the device."""

    wrists = [keypoints.get(9), keypoints.get(10)]
    for wrist in wrists:
        if wrist is None or wrist[2] < 0.20:
            continue
        if wrist[1] <= max(0.35, fallback_box.top + 0.08) and abs(wrist[0] - 0.5) <= 0.22:
            return True
    return False


def _visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    index: int,
    *,
    min_score: float = 0.20,
) -> tuple[float, float, float] | None:
    """Return one keypoint only when its score clears the minimum threshold."""

    joint = keypoints.get(index)
    if joint is None or joint[2] < min_score:
        return None
    return joint


def _best_visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    indices: tuple[int, ...],
    *,
    min_score: float = 0.20,
) -> tuple[float, float, float] | None:
    """Return the highest-confidence visible joint from one candidate set."""

    visible = [
        joint
        for index in indices
        if (joint := _visible_joint(keypoints, index, min_score=min_score)) is not None
    ]
    if not visible:
        return None
    return max(visible, key=lambda item: item[2])


def _strong_keypoint_count(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    min_score: float = 0.20,
) -> int:
    """Count how many keypoints are strong enough to support coarse inference."""

    return sum(1 for _, _, score in keypoints.values() if score >= min_score)


def _matches_wave_arm(
    *,
    shoulder: tuple[float, float, float] | None,
    elbow: tuple[float, float, float] | None,
    wrist: tuple[float, float, float] | None,
    side: str,
) -> bool:
    """Return whether one arm geometry looks like a simple raised wave pose."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    lateral = wrist[0] - shoulder[0]
    if side == "left":
        lateral *= -1.0
    return (
        wrist[1] < shoulder[1] - 0.05
        and lateral >= 0.12
        and elbow[1] > wrist[1]
        and elbow[1] <= shoulder[1] + 0.18
    )


def _matches_vertical_arm(
    *,
    shoulder: tuple[float, float, float] | None,
    elbow: tuple[float, float, float] | None,
    wrist: tuple[float, float, float] | None,
    shoulder_span: float,
) -> bool:
    """Return whether one arm is held mostly vertical."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    x_tolerance = max(0.05, shoulder_span * 0.18)
    return (
        abs(wrist[0] - elbow[0]) <= x_tolerance
        and abs(elbow[0] - shoulder[0]) <= x_tolerance
        and wrist[1] < elbow[1] < (shoulder[1] + 0.20)
    )


def _matches_horizontal_arm_toward_center(
    *,
    shoulder: tuple[float, float, float] | None,
    elbow: tuple[float, float, float] | None,
    wrist: tuple[float, float, float] | None,
    shoulder_center_x: float,
    shoulder_y: float,
    side: str,
) -> bool:
    """Return whether one forearm is held roughly horizontal toward the torso center."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    toward_center = wrist[0] > shoulder[0] if side == "left" else wrist[0] < shoulder[0]
    return (
        toward_center
        and abs(wrist[1] - elbow[1]) <= 0.08
        and abs(elbow[1] - shoulder_y) <= 0.16
        and abs(wrist[0] - shoulder_center_x) <= 0.16
    )


def _support_pose_confidence(
    raw_score: float,
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> float:
    """Normalize pose confidence using keypoint support instead of raw score alone."""

    normalized_raw = _clamp_ratio(raw_score, default=0.0)
    if normalized_raw <= 0.0:
        return 0.0

    strong_fraction = _strong_keypoint_count(keypoints) / float(_KEYPOINT_COUNT)
    shoulders = sum(1 for index in (5, 6) if _visible_joint(keypoints, index) is not None)
    hips = sum(1 for index in (11, 12) if _visible_joint(keypoints, index) is not None)
    wrists = sum(1 for index in (9, 10) if _visible_joint(keypoints, index) is not None)
    legs = sum(1 for index in (13, 14, 15, 16) if _visible_joint(keypoints, index) is not None)
    face = sum(1 for index in (0, 1, 2) if _visible_joint(keypoints, index) is not None)

    structure_support = 0.0
    if shoulders > 0 and hips > 0:
        structure_support += 0.35
    elif shoulders > 0 or hips > 0:
        structure_support += 0.20
    if legs >= 2:
        structure_support += 0.20
    if wrists >= 1:
        structure_support += 0.10
    if face >= 1:
        structure_support += 0.10
    if fallback_box.height >= 0.45 and fallback_box.width <= 0.60:
        structure_support += 0.15

    support_score = max(strong_fraction, min(1.0, structure_support))
    return round(min(normalized_raw, support_score), 3)


def _classify_gesture(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    attention_score: float,
    fallback_box: AICameraBox,
) -> tuple[AICameraGestureEvent, float | None]:
    """Return one conservative coarse-arm gesture classification."""

    left_shoulder = _visible_joint(keypoints, 5)
    right_shoulder = _visible_joint(keypoints, 6)
    left_elbow = _visible_joint(keypoints, 7)
    right_elbow = _visible_joint(keypoints, 8)
    left_wrist = _visible_joint(keypoints, 9)
    right_wrist = _visible_joint(keypoints, 10)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        shoulder_span = max(0.08, abs(right_shoulder[0] - left_shoulder[0]))
        chest_bottom = shoulder_y + max(0.18, fallback_box.height * 0.24)
        crossed_margin = max(0.03, shoulder_span * 0.08)
        horizontal_outreach = max(0.12, shoulder_span * 0.28)

        if (
            left_wrist is not None
            and right_wrist is not None
            and shoulder_y - 0.04 <= left_wrist[1] <= chest_bottom
            and shoulder_y - 0.04 <= right_wrist[1] <= chest_bottom
            and left_wrist[0] >= shoulder_center_x + crossed_margin
            and right_wrist[0] <= shoulder_center_x - crossed_margin
        ):
            return AICameraGestureEvent.ARMS_CROSSED, round(
                _clamp_ratio(0.62 + 0.18 * attention_score, default=0.62),
                3,
            )

        if (
            left_wrist is not None
            and right_wrist is not None
            and left_wrist[0] <= left_shoulder[0] - horizontal_outreach
            and right_wrist[0] >= right_shoulder[0] + horizontal_outreach
            and abs(left_wrist[1] - left_shoulder[1]) <= 0.18
            and abs(right_wrist[1] - right_shoulder[1]) <= 0.18
        ):
            return AICameraGestureEvent.TWO_HAND_DISMISS, round(
                _clamp_ratio(0.66 + 0.18 * attention_score, default=0.66),
                3,
            )

        if (
            _matches_horizontal_arm_toward_center(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_center_x=shoulder_center_x,
                shoulder_y=shoulder_y,
                side="left",
            )
            and _matches_vertical_arm(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_span=shoulder_span,
            )
        ) or (
            _matches_horizontal_arm_toward_center(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_center_x=shoulder_center_x,
                shoulder_y=shoulder_y,
                side="right",
            )
            and _matches_vertical_arm(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_span=shoulder_span,
            )
        ):
            return AICameraGestureEvent.TIMEOUT_T, round(
                _clamp_ratio(0.64 + 0.18 * attention_score, default=0.64),
                3,
            )

    if _matches_wave_arm(shoulder=left_shoulder, elbow=left_elbow, wrist=left_wrist, side="left") or _matches_wave_arm(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
    ):
        return AICameraGestureEvent.WAVE, round(_clamp_ratio(0.60 + 0.20 * attention_score, default=0.60), 3)

    if left_shoulder and left_wrist:
        if left_wrist[1] < left_shoulder[1] - 0.08 and abs(left_wrist[0] - left_shoulder[0]) <= 0.16:
            return AICameraGestureEvent.STOP, round(_clamp_ratio(0.7 + 0.3 * attention_score, default=0.7), 3)
        if left_wrist[0] < left_shoulder[0] - 0.18 and abs(left_wrist[1] - left_shoulder[1]) <= 0.16:
            return AICameraGestureEvent.DISMISS, round(_clamp_ratio(0.6 + 0.2 * attention_score, default=0.6), 3)
        if (
            left_elbow
            and abs(left_wrist[0] - left_shoulder[0]) <= 0.08
            and left_wrist[1] <= left_shoulder[1]
            and left_elbow[1] > left_wrist[1]
        ):
            return AICameraGestureEvent.CONFIRM, round(_clamp_ratio(0.55 + 0.2 * attention_score, default=0.55), 3)

    if right_shoulder and right_wrist:
        if right_wrist[1] < right_shoulder[1] - 0.08 and abs(right_wrist[0] - right_shoulder[0]) <= 0.16:
            return AICameraGestureEvent.STOP, round(_clamp_ratio(0.7 + 0.3 * attention_score, default=0.7), 3)
        if right_wrist[0] > right_shoulder[0] + 0.18 and abs(right_wrist[1] - right_shoulder[1]) <= 0.16:
            return AICameraGestureEvent.DISMISS, round(_clamp_ratio(0.6 + 0.2 * attention_score, default=0.6), 3)
        if (
            right_elbow
            and abs(right_wrist[0] - right_shoulder[0]) <= 0.08
            and right_wrist[1] <= right_shoulder[1]
            and right_elbow[1] > right_wrist[1]
        ):
            return AICameraGestureEvent.CONFIRM, round(_clamp_ratio(0.55 + 0.2 * attention_score, default=0.55), 3)

    wrists = [
        joint
        for joint in (
            _visible_joint(keypoints, 9),
            _visible_joint(keypoints, 10),
        )
        if joint is not None
    ]
    stop_top = fallback_box.top + max(0.10, fallback_box.height * 0.20)
    dismiss_offset = max(0.18, fallback_box.width * 0.35)
    stop_offset = max(0.14, fallback_box.width * 0.28)
    for wrist in wrists:
        wrist_x, wrist_y, _ = wrist
        center_offset = abs(wrist_x - fallback_box.center_x)
        if wrist_y <= stop_top and center_offset <= stop_offset:
            return AICameraGestureEvent.STOP, round(_clamp_ratio(0.58 + 0.25 * attention_score, default=0.58), 3)
        if (
            wrist_y <= fallback_box.top + fallback_box.height * 0.65
            and center_offset >= dismiss_offset
        ):
            return AICameraGestureEvent.DISMISS, round(_clamp_ratio(0.5 + 0.2 * attention_score, default=0.5), 3)

    return AICameraGestureEvent.NONE, None


def _classify_body_pose(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
) -> AICameraBodyPose:
    """Return one coarse, conservative body pose classification."""

    left_shoulder = _visible_joint(keypoints, 5)
    right_shoulder = _visible_joint(keypoints, 6)
    left_hip = _visible_joint(keypoints, 11)
    right_hip = _visible_joint(keypoints, 12)
    left_knee = _visible_joint(keypoints, 13)
    right_knee = _visible_joint(keypoints, 14)
    left_ankle = _visible_joint(keypoints, 15)
    right_ankle = _visible_joint(keypoints, 16)
    shoulders_ready = left_shoulder is not None and right_shoulder is not None
    hips_ready = left_hip is not None and right_hip is not None

    if fallback_box.area <= 0.0:
        return AICameraBodyPose.UNKNOWN
    if fallback_box.width >= 0.42 and fallback_box.height <= 0.28 and fallback_box.center_y >= 0.72:
        return AICameraBodyPose.FLOOR
    if fallback_box.width >= max(0.42, fallback_box.height * 1.15):
        return AICameraBodyPose.LYING_LOW
    if shoulders_ready and hips_ready:
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
        torso_vertical = max(0.0, hip_center_y - shoulder_center_y)
        torso_horizontal = max(0.02, abs(hip_center_x - shoulder_center_x))
        torso_ratio = torso_vertical / torso_horizontal
        if torso_ratio <= 0.65:
            return AICameraBodyPose.LYING_LOW if fallback_box.center_y < 0.72 else AICameraBodyPose.FLOOR
        if left_knee and right_knee and min(left_knee[2], right_knee[2]) >= 0.20:
            knee_center_y = (left_knee[1] + right_knee[1]) / 2.0
            if knee_center_y - hip_center_y <= 0.18:
                return AICameraBodyPose.SEATED
        if torso_ratio < 1.1 or fallback_box.center_y > 0.68:
            return AICameraBodyPose.SLUMPED
        return AICameraBodyPose.UPRIGHT
    side_candidates = (
        (left_shoulder, left_hip, left_knee, left_ankle),
        (right_shoulder, right_hip, right_knee, right_ankle),
    )
    best_side = max(
        side_candidates,
        key=lambda item: sum(joint is not None for joint in item),
    )
    side_shoulder, side_hip, side_knee, side_ankle = best_side
    if side_shoulder is not None and side_hip is not None:
        torso_vertical = max(0.0, side_hip[1] - side_shoulder[1])
        torso_horizontal = max(0.03, abs(side_hip[0] - side_shoulder[0]), fallback_box.width * 0.12)
        torso_ratio = torso_vertical / torso_horizontal
        if torso_ratio <= 0.65:
            return AICameraBodyPose.LYING_LOW if fallback_box.center_y < 0.72 else AICameraBodyPose.FLOOR
        if side_knee is not None and (side_knee[1] - side_hip[1]) <= 0.16:
            return AICameraBodyPose.SEATED
        if side_ankle is not None and (side_ankle[1] - side_hip[1]) >= 0.18:
            if torso_ratio < 0.95 or fallback_box.center_y > 0.68:
                return AICameraBodyPose.SLUMPED
            return AICameraBodyPose.UPRIGHT
        if torso_ratio < 0.95:
            return AICameraBodyPose.SLUMPED
        if fallback_box.height >= 0.68 and fallback_box.width <= 0.46:
            return AICameraBodyPose.UPRIGHT
    if side_hip is not None and side_knee is not None:
        if (side_knee[1] - side_hip[1]) <= 0.15:
            return AICameraBodyPose.SEATED
        if side_ankle is not None and (side_ankle[1] - side_hip[1]) >= 0.20 and fallback_box.height >= 0.70:
            return AICameraBodyPose.UPRIGHT
    if fallback_box.height >= 0.78 and fallback_box.width <= 0.42 and fallback_box.center_y <= 0.62:
        return AICameraBodyPose.UPRIGHT
    if (
        fallback_box.height >= 0.42
        and fallback_box.height <= 0.72
        and fallback_box.width <= 0.56
        and fallback_box.center_y >= 0.56
    ):
        return AICameraBodyPose.SEATED
    return AICameraBodyPose.UNKNOWN


def _infer_motion_state(
    *,
    previous_box: AICameraBox | None,
    current_box: AICameraBox | None,
    previous_observed_at: float | None,
    current_observed_at: float,
    previous_person_count: int,
    current_person_count: int,
) -> tuple[AICameraMotionState, float | None]:
    """Infer one coarse motion state from recent primary-person box deltas."""

    if (
        previous_box is None
        or current_box is None
        or previous_observed_at is None
        or previous_person_count != 1
        or current_person_count != 1
    ):
        return AICameraMotionState.UNKNOWN, None

    delta_t = max(0.0, float(current_observed_at) - float(previous_observed_at))
    if delta_t < _MOTION_MIN_DELTA_S or delta_t > _MOTION_UNKNOWN_MAX_GAP_S:
        return AICameraMotionState.UNKNOWN, None

    delta_x = current_box.center_x - previous_box.center_x
    delta_y = current_box.center_y - previous_box.center_y
    center_distance = math.hypot(delta_x, delta_y)
    center_speed = center_distance / max(delta_t, _MOTION_MIN_DELTA_S)
    area_delta = current_box.area - previous_box.area
    height_delta = current_box.height - previous_box.height
    scale_strength = max(abs(area_delta), abs(height_delta))
    confidence = round(
        _clamp_ratio(max(center_speed * 5.0, scale_strength * 3.4), default=0.0),
        3,
    )

    if abs(area_delta) >= 0.06 and abs(delta_x) <= 0.10:
        if area_delta > 0.0 and height_delta >= -0.01:
            return AICameraMotionState.APPROACHING, max(confidence, 0.56)
        if area_delta < 0.0 and height_delta <= 0.01:
            return AICameraMotionState.LEAVING, max(confidence, 0.56)

    if center_speed >= 0.08 or (abs(delta_x) >= 0.08 and scale_strength <= 0.08):
        return AICameraMotionState.WALKING, max(confidence, 0.54)

    if center_distance <= 0.035 and scale_strength <= 0.04:
        return AICameraMotionState.STILL, max(0.52, round(0.52 + (0.04 - scale_strength), 3))

    if abs(area_delta) >= 0.04:
        return (
            (AICameraMotionState.APPROACHING if area_delta > 0.0 else AICameraMotionState.LEAVING),
            max(confidence, 0.5),
        )
    return AICameraMotionState.STILL, max(confidence, 0.45)


__all__ = [
    "AICameraAdapterConfig",
    "AICameraBodyPose",
    "AICameraBox",
    "AICameraGestureEvent",
    "AICameraFineHandGesture",
    "AICameraMotionState",
    "AICameraObjectDetection",
    "AICameraObservation",
    "AICameraZone",
    "LocalAICameraAdapter",
]
