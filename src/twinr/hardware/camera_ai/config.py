"""Store bounded configuration for Twinr's decomposed local AI-camera stack."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path, PurePath
from typing import cast

from twinr.agent.base_agent.config import TwinrConfig


DEFAULT_MEDIAPIPE_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MEDIAPIPE_GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)

_DEFAULT_DETECTION_NETWORK = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
_DEFAULT_POSE_NETWORK = "/usr/share/imx500-models/imx500_network_posenet.rpk"
_DEFAULT_POSE_BACKEND = "mediapipe"
_DEFAULT_MEDIAPIPE_POSE_MODEL = "state/mediapipe/models/pose_landmarker_full.task"
_DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL = "state/mediapipe/models/hand_landmarker.task"
_DEFAULT_MEDIAPIPE_GESTURE_MODEL = "state/mediapipe/models/gesture_recognizer.task"
_DEFAULT_MEDIAPIPE_CUSTOM_GESTURE_MODEL = "state/mediapipe/models/custom_gesture.task"
_DEFAULT_MAIN_SIZE = (640, 480)
_DEFAULT_FRAME_RATE = 15
_DEFAULT_LOW_LIGHT_FRAME_RATE = 6
_DEFAULT_LOCK_TIMEOUT_S = 5.0
_DEFAULT_STARTUP_WARMUP_S = 0.8
_DEFAULT_METADATA_WAIT_S = 0.75
_DEFAULT_LOW_LIGHT_LUX_THRESHOLD = 1.2
_DEFAULT_LOW_LIGHT_RECOVER_LUX_THRESHOLD = 2.2
_DEFAULT_LOW_LIGHT_MANUAL_EXPOSURE_RATIO = 0.90
_DEFAULT_LOW_LIGHT_MANUAL_ANALOGUE_GAIN = 8.0
_DEFAULT_LOW_LIGHT_AUTO_EXPOSURE_CAP_RATIO = 0.60
_DEFAULT_GESTURE_CANDIDATE_CAPTURE_DIR = "artifacts/tmp/gesture_candidate_captures"
_DEFAULT_GESTURE_CANDIDATE_CAPTURE_COOLDOWN_S = 1.5
_DEFAULT_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES = 24
_DEFAULT_PERSON_CONFIDENCE = 0.40
_DEFAULT_OBJECT_CONFIDENCE = 0.55
_DEFAULT_PERSON_NEAR_AREA = 0.20
_DEFAULT_PERSON_NEAR_HEIGHT = 0.55
_DEFAULT_OBJECT_NEAR_AREA = 0.08
_DEFAULT_ATTENTION_SCORE_THRESHOLD = 0.62
_DEFAULT_ENGAGED_SCORE_THRESHOLD = 0.45
_DEFAULT_POSE_CONFIDENCE = 0.30
_DEFAULT_POSE_REFRESH_S = 0.75
_DEFAULT_SEQUENCE_WINDOW_S = 0.55
_DEFAULT_SEQUENCE_MIN_FRAMES = 3

_ALLOWED_POSE_BACKENDS = frozenset({"imx500", "mediapipe"})  # AUDIT-FIX(#3): Limit pose backends to the supported runtime choices.
_ALLOWED_RPK_SUFFIXES = (".rpk",)  # AUDIT-FIX(#2): Restrict network assets to expected local IMX500 package files.
_ALLOWED_TASK_SUFFIXES = (".task",)  # AUDIT-FIX(#2): Restrict MediaPipe assets to expected local task bundle files.

_MIN_CAMERA_WIDTH = 160  # AUDIT-FIX(#1): Keep camera sizing inside Pi-safe, device-realistic limits.
_MAX_CAMERA_WIDTH = 1920  # AUDIT-FIX(#1): Keep camera sizing inside Pi-safe, device-realistic limits.
_MIN_CAMERA_HEIGHT = 120  # AUDIT-FIX(#1): Keep camera sizing inside Pi-safe, device-realistic limits.
_MAX_CAMERA_HEIGHT = 1080  # AUDIT-FIX(#1): Keep camera sizing inside Pi-safe, device-realistic limits.
_MIN_FRAME_RATE = 1  # AUDIT-FIX(#1): Avoid zero/negative FPS and cap runaway CPU usage on Pi 4.
_MAX_FRAME_RATE = 30  # AUDIT-FIX(#1): Avoid zero/negative FPS and cap runaway CPU usage on Pi 4.
_MAX_LUX_THRESHOLD = 1000.0  # AUDIT-FIX(#12): Keep low-light thresholds finite and physically plausible.
_MIN_NUM_HANDS = 1  # AUDIT-FIX(#1): Bound hand tracking to supported, realistic values.
_MAX_NUM_HANDS = 2  # AUDIT-FIX(#1): Bound hand tracking to supported, realistic values.
_MIN_LOCK_TIMEOUT_S = 0.1  # AUDIT-FIX(#1): Prevent effectively disabled or pathological lock timing.
_MAX_LOCK_TIMEOUT_S = 60.0  # AUDIT-FIX(#1): Prevent effectively disabled or pathological lock timing.
_MAX_STARTUP_WARMUP_S = 30.0  # AUDIT-FIX(#1): Prevent unbounded boot delays from malformed config.
_MIN_METADATA_WAIT_S = 0.1  # AUDIT-FIX(#1): Prevent immediate timeout loops.
_MAX_METADATA_WAIT_S = 30.0  # AUDIT-FIX(#1): Prevent excessive blocking waits.
_MAX_POSE_REFRESH_S = 300.0  # AUDIT-FIX(#1): Prevent stale pose state for arbitrarily long periods.
_MIN_SEQUENCE_WINDOW_S = 0.1  # AUDIT-FIX(#1): Keep temporal windows finite and usable.
_MAX_SEQUENCE_WINDOW_S = 10.0  # AUDIT-FIX(#1): Keep temporal windows finite and usable.
_MIN_SEQUENCE_MIN_FRAMES = 2  # AUDIT-FIX(#1): Require at least a minimally meaningful temporal sequence.
_MAX_SEQUENCE_MIN_FRAMES = 120  # AUDIT-FIX(#1): Prevent impossible gesture windows and runaway buffering.
_MAX_GESTURE_CANDIDATE_CAPTURE_COOLDOWN_S = 300.0  # AUDIT-FIX(#13): Keep debug capture cooldown bounded on the Pi.
_MIN_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES = 1  # AUDIT-FIX(#13): Always keep at least one QA frame slot when enabled.
_MAX_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES = 128  # AUDIT-FIX(#13): Prevent unbounded artifact growth from QA capture.
_MIN_ROI_CANDIDATES = 1  # AUDIT-FIX(#11): Keep ROI fan-out bounded on the Pi.
_MAX_ROI_CANDIDATES = 6  # AUDIT-FIX(#11): Bound per-frame MediaPipe hand ROI work.
_ALLOWED_MEDIAPIPE_LIVE_GESTURE_MODES = frozenset({"video", "live_stream"})
_DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S = 1.0
_MAX_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S = 30.0


@dataclass(frozen=True, slots=True)
class AICameraAdapterConfig:
    """Store bounded runtime settings for the local IMX500 adapter."""

    detection_network_path: str = _DEFAULT_DETECTION_NETWORK
    pose_network_path: str = _DEFAULT_POSE_NETWORK
    pose_backend: str = _DEFAULT_POSE_BACKEND
    mediapipe_pose_model_path: str = _DEFAULT_MEDIAPIPE_POSE_MODEL
    mediapipe_hand_landmarker_model_path: str = _DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL
    mediapipe_gesture_model_path: str = _DEFAULT_MEDIAPIPE_GESTURE_MODEL
    mediapipe_custom_gesture_model_path: str | None = None
    mediapipe_num_hands: int = 2
    main_size: tuple[int, int] = _DEFAULT_MAIN_SIZE
    frame_rate: int = _DEFAULT_FRAME_RATE
    low_light_frame_rate: int = _DEFAULT_LOW_LIGHT_FRAME_RATE
    lock_timeout_s: float = _DEFAULT_LOCK_TIMEOUT_S
    startup_warmup_s: float = _DEFAULT_STARTUP_WARMUP_S
    metadata_wait_s: float = _DEFAULT_METADATA_WAIT_S
    low_light_lux_threshold: float = _DEFAULT_LOW_LIGHT_LUX_THRESHOLD
    low_light_recover_lux_threshold: float = _DEFAULT_LOW_LIGHT_RECOVER_LUX_THRESHOLD
    low_light_manual_exposure_ratio: float = _DEFAULT_LOW_LIGHT_MANUAL_EXPOSURE_RATIO
    low_light_manual_analogue_gain: float = _DEFAULT_LOW_LIGHT_MANUAL_ANALOGUE_GAIN
    low_light_auto_exposure_cap_ratio: float = _DEFAULT_LOW_LIGHT_AUTO_EXPOSURE_CAP_RATIO
    gesture_candidate_capture_dir: str = _DEFAULT_GESTURE_CANDIDATE_CAPTURE_DIR
    gesture_candidate_capture_cooldown_s: float = _DEFAULT_GESTURE_CANDIDATE_CAPTURE_COOLDOWN_S
    gesture_candidate_capture_max_images: int = _DEFAULT_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES
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
    builtin_gesture_min_score: float = 0.35
    custom_gesture_min_score: float = 0.45
    min_hand_detection_confidence: float = 0.35
    min_hand_presence_confidence: float = 0.35
    min_hand_tracking_confidence: float = 0.35
    max_roi_candidates: int = 4
    primary_person_roi_padding: float = 0.18
    primary_person_upper_body_ratio: float = 0.78
    wrist_roi_scale: float = 0.34
    live_pending_result_timeout_s: float = 1.0
    mediapipe_native_heap_trim_interval_s: float = _DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S

    def __post_init__(self) -> None:
        """Enforce safe bounds even for direct dataclass construction."""

        # AUDIT-FIX(#2): Normalize all file path inputs before later file-open code sees them.
        object.__setattr__(
            self,
            "detection_network_path",
            _coerce_local_path(
                self.detection_network_path,
                default=_DEFAULT_DETECTION_NETWORK,
                expected_suffixes=_ALLOWED_RPK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "pose_network_path",
            _coerce_local_path(
                self.pose_network_path,
                default=_DEFAULT_POSE_NETWORK,
                expected_suffixes=_ALLOWED_RPK_SUFFIXES,
            ),
        )
        # AUDIT-FIX(#3): Fall back to a supported backend instead of propagating typos.
        object.__setattr__(
            self,
            "pose_backend",
            _coerce_choice(
                self.pose_backend,
                default=_DEFAULT_POSE_BACKEND,
                allowed=_ALLOWED_POSE_BACKENDS,
            ),
        )
        object.__setattr__(
            self,
            "mediapipe_pose_model_path",
            _coerce_local_path(
                self.mediapipe_pose_model_path,
                default=_DEFAULT_MEDIAPIPE_POSE_MODEL,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "mediapipe_hand_landmarker_model_path",
            _coerce_local_path(
                self.mediapipe_hand_landmarker_model_path,
                default=_DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "mediapipe_gesture_model_path",
            _coerce_local_path(
                self.mediapipe_gesture_model_path,
                default=_DEFAULT_MEDIAPIPE_GESTURE_MODEL,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "mediapipe_custom_gesture_model_path",
            _coerce_local_path(
                self.mediapipe_custom_gesture_model_path,
                default=None,
                allow_none=True,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )

        # AUDIT-FIX(#1): Enforce actual runtime bounds so "bounded" config cannot destabilize the Pi.
        object.__setattr__(
            self,
            "mediapipe_num_hands",
            _coerce_bounded_int(
                self.mediapipe_num_hands,
                default=2,
                minimum=_MIN_NUM_HANDS,
                maximum=_MAX_NUM_HANDS,
            ),
        )
        object.__setattr__(self, "main_size", _coerce_main_size(self.main_size, default=_DEFAULT_MAIN_SIZE))

        frame_rate = _coerce_bounded_int(
            self.frame_rate,
            default=_DEFAULT_FRAME_RATE,
            minimum=_MIN_FRAME_RATE,
            maximum=_MAX_FRAME_RATE,
        )
        object.__setattr__(self, "frame_rate", frame_rate)
        object.__setattr__(
            self,
            "low_light_frame_rate",
            min(
                frame_rate,
                _coerce_bounded_int(
                    self.low_light_frame_rate,
                    default=_DEFAULT_LOW_LIGHT_FRAME_RATE,
                    minimum=_MIN_FRAME_RATE,
                    maximum=_MAX_FRAME_RATE,
                ),
            ),
        )

        object.__setattr__(
            self,
            "lock_timeout_s",
            _coerce_bounded_float(
                self.lock_timeout_s,
                default=_DEFAULT_LOCK_TIMEOUT_S,
                minimum=_MIN_LOCK_TIMEOUT_S,
                maximum=_MAX_LOCK_TIMEOUT_S,
            ),
        )
        object.__setattr__(
            self,
            "startup_warmup_s",
            _coerce_bounded_float(
                self.startup_warmup_s,
                default=_DEFAULT_STARTUP_WARMUP_S,
                minimum=0.0,
                maximum=_MAX_STARTUP_WARMUP_S,
            ),
        )
        object.__setattr__(
            self,
            "metadata_wait_s",
            _coerce_bounded_float(
                self.metadata_wait_s,
                default=_DEFAULT_METADATA_WAIT_S,
                minimum=_MIN_METADATA_WAIT_S,
                maximum=_MAX_METADATA_WAIT_S,
            ),
        )
        low_light_lux_threshold = _coerce_bounded_float(
            self.low_light_lux_threshold,
            default=_DEFAULT_LOW_LIGHT_LUX_THRESHOLD,
            minimum=0.0,
            maximum=_MAX_LUX_THRESHOLD,
        )
        object.__setattr__(self, "low_light_lux_threshold", low_light_lux_threshold)
        object.__setattr__(
            self,
            "low_light_recover_lux_threshold",
            max(
                low_light_lux_threshold,
                _coerce_bounded_float(
                    self.low_light_recover_lux_threshold,
                    default=_DEFAULT_LOW_LIGHT_RECOVER_LUX_THRESHOLD,
                    minimum=0.0,
                    maximum=_MAX_LUX_THRESHOLD,
                ),
            ),
        )
        object.__setattr__(
            self,
            "low_light_manual_exposure_ratio",
            _clamp_ratio(
                self.low_light_manual_exposure_ratio,
                default=_DEFAULT_LOW_LIGHT_MANUAL_EXPOSURE_RATIO,
            ),
        )
        object.__setattr__(
            self,
            "low_light_manual_analogue_gain",
            _coerce_bounded_float(
                self.low_light_manual_analogue_gain,
                default=_DEFAULT_LOW_LIGHT_MANUAL_ANALOGUE_GAIN,
                minimum=1.0,
                maximum=16.0,
            ),
        )
        object.__setattr__(
            self,
            "low_light_auto_exposure_cap_ratio",
            _clamp_ratio(
                self.low_light_auto_exposure_cap_ratio,
                default=_DEFAULT_LOW_LIGHT_AUTO_EXPOSURE_CAP_RATIO,
            ),
        )
        object.__setattr__(
            self,
            "gesture_candidate_capture_dir",
            _coerce_local_path(
                self.gesture_candidate_capture_dir,
                default=_DEFAULT_GESTURE_CANDIDATE_CAPTURE_DIR,
            )
            or _DEFAULT_GESTURE_CANDIDATE_CAPTURE_DIR,
        )
        object.__setattr__(
            self,
            "gesture_candidate_capture_cooldown_s",
            _coerce_bounded_float(
                self.gesture_candidate_capture_cooldown_s,
                default=_DEFAULT_GESTURE_CANDIDATE_CAPTURE_COOLDOWN_S,
                minimum=0.0,
                maximum=_MAX_GESTURE_CANDIDATE_CAPTURE_COOLDOWN_S,
            ),
        )
        object.__setattr__(
            self,
            "gesture_candidate_capture_max_images",
            _coerce_bounded_int(
                self.gesture_candidate_capture_max_images,
                default=_DEFAULT_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES,
                minimum=_MIN_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES,
                maximum=_MAX_GESTURE_CANDIDATE_CAPTURE_MAX_IMAGES,
            ),
        )

        object.__setattr__(
            self,
            "person_confidence_threshold",
            _clamp_ratio(self.person_confidence_threshold, default=_DEFAULT_PERSON_CONFIDENCE),
        )
        object.__setattr__(
            self,
            "object_confidence_threshold",
            _clamp_ratio(self.object_confidence_threshold, default=_DEFAULT_OBJECT_CONFIDENCE),
        )
        object.__setattr__(
            self,
            "person_near_area_threshold",
            _clamp_ratio(self.person_near_area_threshold, default=_DEFAULT_PERSON_NEAR_AREA),
        )
        object.__setattr__(
            self,
            "person_near_height_threshold",
            _clamp_ratio(self.person_near_height_threshold, default=_DEFAULT_PERSON_NEAR_HEIGHT),
        )
        object.__setattr__(
            self,
            "object_near_area_threshold",
            _clamp_ratio(self.object_near_area_threshold, default=_DEFAULT_OBJECT_NEAR_AREA),
        )
        object.__setattr__(
            self,
            "attention_score_threshold",
            _clamp_ratio(self.attention_score_threshold, default=_DEFAULT_ATTENTION_SCORE_THRESHOLD),
        )
        object.__setattr__(
            self,
            "engaged_score_threshold",
            _clamp_ratio(self.engaged_score_threshold, default=_DEFAULT_ENGAGED_SCORE_THRESHOLD),
        )
        object.__setattr__(
            self,
            "pose_confidence_threshold",
            _clamp_ratio(self.pose_confidence_threshold, default=_DEFAULT_POSE_CONFIDENCE),
        )
        object.__setattr__(
            self,
            "builtin_gesture_min_score",
            _clamp_ratio(self.builtin_gesture_min_score, default=0.35),
        )
        object.__setattr__(
            self,
            "custom_gesture_min_score",
            _clamp_ratio(self.custom_gesture_min_score, default=0.45),
        )
        object.__setattr__(
            self,
            "min_hand_detection_confidence",
            _clamp_ratio(self.min_hand_detection_confidence, default=0.35),
        )
        object.__setattr__(
            self,
            "min_hand_presence_confidence",
            _clamp_ratio(self.min_hand_presence_confidence, default=0.35),
        )
        object.__setattr__(
            self,
            "min_hand_tracking_confidence",
            _clamp_ratio(self.min_hand_tracking_confidence, default=0.35),
        )
        object.__setattr__(
            self,
            "max_roi_candidates",
            _coerce_bounded_int(
                self.max_roi_candidates,
                default=4,
                minimum=_MIN_ROI_CANDIDATES,
                maximum=_MAX_ROI_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "primary_person_roi_padding",
            _clamp_ratio(self.primary_person_roi_padding, default=0.18),
        )
        object.__setattr__(
            self,
            "primary_person_upper_body_ratio",
            _clamp_ratio(self.primary_person_upper_body_ratio, default=0.78),
        )
        object.__setattr__(
            self,
            "wrist_roi_scale",
            _clamp_ratio(self.wrist_roi_scale, default=0.34),
        )
        object.__setattr__(
            self,
            "live_pending_result_timeout_s",
            _coerce_bounded_float(
                self.live_pending_result_timeout_s,
                default=1.0,
                minimum=0.25,
                maximum=5.0,
            ),
        )
        object.__setattr__(
            self,
            "mediapipe_native_heap_trim_interval_s",
            _coerce_bounded_float(
                self.mediapipe_native_heap_trim_interval_s,
                default=_coerce_bounded_float(
                    os.getenv(
                        "TWINR_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S",
                        _DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                    ),
                    default=_DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                    minimum=0.0,
                    maximum=_MAX_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                ),
                minimum=0.0,
                maximum=_MAX_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
            ),
        )
        object.__setattr__(
            self,
            "pose_refresh_s",
            _coerce_bounded_float(
                self.pose_refresh_s,
                default=_DEFAULT_POSE_REFRESH_S,
                minimum=0.0,
                maximum=_MAX_POSE_REFRESH_S,
            ),
        )

        sequence_min_frames = _coerce_bounded_int(
            self.sequence_min_frames,
            default=_DEFAULT_SEQUENCE_MIN_FRAMES,
            minimum=_MIN_SEQUENCE_MIN_FRAMES,
            maximum=_MAX_SEQUENCE_MIN_FRAMES,
        )
        minimum_feasible_window_s = min(_MAX_SEQUENCE_WINDOW_S, sequence_min_frames / frame_rate)
        sequence_window_s = _coerce_bounded_float(
            self.sequence_window_s,
            default=_DEFAULT_SEQUENCE_WINDOW_S,
            minimum=_MIN_SEQUENCE_WINDOW_S,
            maximum=_MAX_SEQUENCE_WINDOW_S,
        )
        if sequence_window_s < minimum_feasible_window_s:
            sequence_window_s = minimum_feasible_window_s
        max_feasible_frames = _max_sequence_min_frames(
            sequence_window_s=sequence_window_s,
            frame_rate=frame_rate,
        )
        object.__setattr__(self, "sequence_window_s", sequence_window_s)
        object.__setattr__(self, "sequence_min_frames", min(sequence_min_frames, max_feasible_frames))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AICameraAdapterConfig":
        """Build one local AI-camera config from ``TwinrConfig``."""

        # AUDIT-FIX(#4): Read config attributes defensively and let __post_init__ normalize malformed values.
        return cls(
            detection_network_path=_safe_getattr(
                config,
                "proactive_local_camera_detection_network_path",
                _DEFAULT_DETECTION_NETWORK,
            ),
            pose_network_path=_safe_getattr(
                config,
                "proactive_local_camera_pose_network_path",
                _DEFAULT_POSE_NETWORK,
            ),
            pose_backend=_safe_getattr(
                config,
                "proactive_local_camera_pose_backend",
                _DEFAULT_POSE_BACKEND,
            ),
            mediapipe_pose_model_path=_safe_getattr(
                config,
                "proactive_local_camera_mediapipe_pose_model_path",
                _DEFAULT_MEDIAPIPE_POSE_MODEL,
            ),
            mediapipe_hand_landmarker_model_path=_safe_getattr(
                config,
                "proactive_local_camera_mediapipe_hand_landmarker_model_path",
                _DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL,
            ),
            mediapipe_gesture_model_path=_safe_getattr(
                config,
                "proactive_local_camera_mediapipe_gesture_model_path",
                _DEFAULT_MEDIAPIPE_GESTURE_MODEL,
            ),
            mediapipe_custom_gesture_model_path=(
                _safe_getattr(
                    config,
                    "proactive_local_camera_mediapipe_custom_gesture_model_path",
                    None,
                )
                or _resolve_default_custom_gesture_model_path(config)
            ),
            mediapipe_num_hands=_safe_getattr(
                config,
                "proactive_local_camera_mediapipe_num_hands",
                2,
            ),
            main_size=(
                _safe_getattr(config, "camera_width", _DEFAULT_MAIN_SIZE[0]),
                _safe_getattr(config, "camera_height", _DEFAULT_MAIN_SIZE[1]),
            ),
            frame_rate=_safe_getattr(
                config,
                "proactive_local_camera_frame_rate",
                _DEFAULT_FRAME_RATE,
            ),
            low_light_frame_rate=_safe_getattr(
                config,
                "proactive_local_camera_low_light_frame_rate",
                _DEFAULT_LOW_LIGHT_FRAME_RATE,
            ),
            lock_timeout_s=_safe_getattr(
                config,
                "proactive_local_camera_lock_timeout_s",
                _DEFAULT_LOCK_TIMEOUT_S,
            ),
            startup_warmup_s=_safe_getattr(
                config,
                "proactive_local_camera_startup_warmup_s",
                _DEFAULT_STARTUP_WARMUP_S,
            ),
            metadata_wait_s=_safe_getattr(
                config,
                "proactive_local_camera_metadata_wait_s",
                _DEFAULT_METADATA_WAIT_S,
            ),
            low_light_lux_threshold=_safe_getattr(
                config,
                "proactive_local_camera_low_light_lux_threshold",
                _DEFAULT_LOW_LIGHT_LUX_THRESHOLD,
            ),
            low_light_recover_lux_threshold=_safe_getattr(
                config,
                "proactive_local_camera_low_light_recover_lux_threshold",
                _DEFAULT_LOW_LIGHT_RECOVER_LUX_THRESHOLD,
            ),
            person_confidence_threshold=_safe_getattr(
                config,
                "proactive_local_camera_person_confidence_threshold",
                _DEFAULT_PERSON_CONFIDENCE,
            ),
            object_confidence_threshold=_safe_getattr(
                config,
                "proactive_local_camera_object_confidence_threshold",
                _DEFAULT_OBJECT_CONFIDENCE,
            ),
            person_near_area_threshold=_safe_getattr(
                config,
                "proactive_local_camera_person_near_area_threshold",
                _DEFAULT_PERSON_NEAR_AREA,
            ),
            person_near_height_threshold=_safe_getattr(
                config,
                "proactive_local_camera_person_near_height_threshold",
                _DEFAULT_PERSON_NEAR_HEIGHT,
            ),
            object_near_area_threshold=_safe_getattr(
                config,
                "proactive_local_camera_object_near_area_threshold",
                _DEFAULT_OBJECT_NEAR_AREA,
            ),
            attention_score_threshold=_safe_getattr(
                config,
                "proactive_local_camera_attention_score_threshold",
                _DEFAULT_ATTENTION_SCORE_THRESHOLD,
            ),
            engaged_score_threshold=_safe_getattr(
                config,
                "proactive_local_camera_engaged_score_threshold",
                _DEFAULT_ENGAGED_SCORE_THRESHOLD,
            ),
            pose_confidence_threshold=_safe_getattr(
                config,
                "proactive_local_camera_pose_confidence_threshold",
                _DEFAULT_POSE_CONFIDENCE,
            ),
            pose_refresh_s=_safe_getattr(
                config,
                "proactive_local_camera_pose_refresh_s",
                _DEFAULT_POSE_REFRESH_S,
            ),
            builtin_gesture_min_score=_safe_getattr(
                config,
                "proactive_local_camera_builtin_gesture_min_score",
                0.35,
            ),
            custom_gesture_min_score=_safe_getattr(
                config,
                "proactive_local_camera_custom_gesture_min_score",
                0.45,
            ),
            min_hand_detection_confidence=_safe_getattr(
                config,
                "proactive_local_camera_min_hand_detection_confidence",
                0.35,
            ),
            min_hand_presence_confidence=_safe_getattr(
                config,
                "proactive_local_camera_min_hand_presence_confidence",
                0.35,
            ),
            min_hand_tracking_confidence=_safe_getattr(
                config,
                "proactive_local_camera_min_hand_tracking_confidence",
                0.35,
            ),
            max_roi_candidates=_safe_getattr(
                config,
                "proactive_local_camera_max_roi_candidates",
                4,
            ),
            primary_person_roi_padding=_safe_getattr(
                config,
                "proactive_local_camera_primary_person_roi_padding",
                0.18,
            ),
            primary_person_upper_body_ratio=_safe_getattr(
                config,
                "proactive_local_camera_primary_person_upper_body_ratio",
                0.78,
            ),
            wrist_roi_scale=_safe_getattr(
                config,
                "proactive_local_camera_wrist_roi_scale",
                0.34,
            ),
            mediapipe_native_heap_trim_interval_s=cast(
                float,
                _safe_getattr(
                    config,
                    "proactive_local_camera_mediapipe_native_heap_trim_interval_s",
                    os.getenv(
                        "TWINR_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S",
                        _DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                    ),
                ),
            ),
            sequence_window_s=_safe_getattr(
                config,
                "proactive_local_camera_sequence_window_s",
                _DEFAULT_SEQUENCE_WINDOW_S,
            ),
            sequence_min_frames=_safe_getattr(
                config,
                "proactive_local_camera_sequence_min_frames",
                _DEFAULT_SEQUENCE_MIN_FRAMES,
            ),
        )


@dataclass(frozen=True, slots=True)
class MediaPipeVisionConfig:
    """Store bounded MediaPipe runtime settings for Pi-side inference."""

    pose_model_path: str
    hand_landmarker_model_path: str
    gesture_model_path: str
    custom_gesture_model_path: str | None = None
    num_hands: int = 2
    attention_score_threshold: float = 0.62
    sequence_window_s: float = 1.0
    sequence_min_frames: int = 3
    builtin_gesture_min_score: float = 0.35
    custom_gesture_min_score: float = 0.45
    min_pose_detection_confidence: float = 0.50
    min_pose_presence_confidence: float = 0.50
    min_pose_tracking_confidence: float = 0.50
    min_hand_detection_confidence: float = 0.35
    min_hand_presence_confidence: float = 0.35
    min_hand_tracking_confidence: float = 0.35
    max_roi_candidates: int = 4
    primary_person_roi_padding: float = 0.18
    primary_person_upper_body_ratio: float = 0.78
    wrist_roi_scale: float = 0.34
    live_pending_result_timeout_s: float = 1.0
    live_gesture_mode: str = "video"
    native_heap_trim_interval_s: float = _DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S

    def __post_init__(self) -> None:
        """Enforce safe bounds even for direct dataclass construction."""

        # AUDIT-FIX(#2): Normalize model paths so blank/suspicious values fall back to known-safe local assets.
        object.__setattr__(
            self,
            "pose_model_path",
            _coerce_local_path(
                self.pose_model_path,
                default=_DEFAULT_MEDIAPIPE_POSE_MODEL,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "hand_landmarker_model_path",
            _coerce_local_path(
                self.hand_landmarker_model_path,
                default=_DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "gesture_model_path",
            _coerce_local_path(
                self.gesture_model_path,
                default=_DEFAULT_MEDIAPIPE_GESTURE_MODEL,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )
        object.__setattr__(
            self,
            "custom_gesture_model_path",
            _coerce_local_path(
                self.custom_gesture_model_path,
                default=None,
                allow_none=True,
                expected_suffixes=_ALLOWED_TASK_SUFFIXES,
            ),
        )

        # AUDIT-FIX(#1): Keep MediaPipe runtime thresholds and temporal windows inside safe operating ranges.
        object.__setattr__(
            self,
            "num_hands",
            _coerce_bounded_int(
                self.num_hands,
                default=2,
                minimum=_MIN_NUM_HANDS,
                maximum=_MAX_NUM_HANDS,
            ),
        )
        object.__setattr__(
            self,
            "attention_score_threshold",
            _clamp_ratio(self.attention_score_threshold, default=_DEFAULT_ATTENTION_SCORE_THRESHOLD),
        )
        object.__setattr__(
            self,
            "sequence_window_s",
            _coerce_bounded_float(
                self.sequence_window_s,
                default=_DEFAULT_SEQUENCE_WINDOW_S,
                minimum=_MIN_SEQUENCE_WINDOW_S,
                maximum=_MAX_SEQUENCE_WINDOW_S,
            ),
        )
        object.__setattr__(
            self,
            "sequence_min_frames",
            _coerce_bounded_int(
                self.sequence_min_frames,
                default=_DEFAULT_SEQUENCE_MIN_FRAMES,
                minimum=_MIN_SEQUENCE_MIN_FRAMES,
                maximum=_MAX_SEQUENCE_MIN_FRAMES,
            ),
        )
        object.__setattr__(
            self,
            "builtin_gesture_min_score",
            _clamp_ratio(self.builtin_gesture_min_score, default=0.35),
        )
        object.__setattr__(
            self,
            "custom_gesture_min_score",
            _clamp_ratio(self.custom_gesture_min_score, default=0.45),
        )
        object.__setattr__(
            self,
            "min_pose_detection_confidence",
            _clamp_ratio(self.min_pose_detection_confidence, default=0.50),
        )
        object.__setattr__(
            self,
            "min_pose_presence_confidence",
            _clamp_ratio(self.min_pose_presence_confidence, default=0.50),
        )
        object.__setattr__(
            self,
            "min_pose_tracking_confidence",
            _clamp_ratio(self.min_pose_tracking_confidence, default=0.50),
        )
        object.__setattr__(
            self,
            "min_hand_detection_confidence",
            _clamp_ratio(self.min_hand_detection_confidence, default=0.35),
        )
        object.__setattr__(
            self,
            "min_hand_presence_confidence",
            _clamp_ratio(self.min_hand_presence_confidence, default=0.35),
        )
        object.__setattr__(
            self,
            "min_hand_tracking_confidence",
            _clamp_ratio(self.min_hand_tracking_confidence, default=0.35),
        )
        object.__setattr__(
            self,
            "max_roi_candidates",
            _coerce_bounded_int(
                self.max_roi_candidates,
                default=4,
                minimum=_MIN_ROI_CANDIDATES,
                maximum=_MAX_ROI_CANDIDATES,
            ),
        )
        object.__setattr__(
            self,
            "primary_person_roi_padding",
            _clamp_ratio(self.primary_person_roi_padding, default=0.18),
        )
        object.__setattr__(
            self,
            "primary_person_upper_body_ratio",
            _clamp_ratio(self.primary_person_upper_body_ratio, default=0.78),
        )
        object.__setattr__(
            self,
            "wrist_roi_scale",
            _clamp_ratio(self.wrist_roi_scale, default=0.34),
        )
        object.__setattr__(
            self,
            "live_pending_result_timeout_s",
            _coerce_bounded_float(
                self.live_pending_result_timeout_s,
                default=1.0,
                minimum=0.25,
                maximum=5.0,
            ),
        )
        object.__setattr__(
            self,
            "live_gesture_mode",
            _coerce_choice(
                self.live_gesture_mode,
                default=os.getenv("TWINR_MEDIAPIPE_LIVE_GESTURE_MODE", "video"),
                allowed=_ALLOWED_MEDIAPIPE_LIVE_GESTURE_MODES,
            ),
        )
        object.__setattr__(
            self,
            "native_heap_trim_interval_s",
            _coerce_bounded_float(
                self.native_heap_trim_interval_s,
                default=_coerce_bounded_float(
                    os.getenv(
                        "TWINR_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S",
                        _DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                    ),
                    default=_DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                    minimum=0.0,
                    maximum=_MAX_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                ),
                minimum=0.0,
                maximum=_MAX_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
            ),
        )

    @classmethod
    def from_ai_camera_config(cls, config: object) -> "MediaPipeVisionConfig":
        """Build one MediaPipe config from ``AICameraAdapterConfig``-like input."""

        # AUDIT-FIX(#4): Avoid raw int() conversion, keep blank paths from propagating, and make temporal settings feasible.
        frame_rate = _coerce_bounded_int(
            _safe_getattr(config, "frame_rate", _DEFAULT_FRAME_RATE),
            default=_DEFAULT_FRAME_RATE,
            minimum=_MIN_FRAME_RATE,
            maximum=_MAX_FRAME_RATE,
        )
        requested_sequence_min_frames = _coerce_bounded_int(
            _safe_getattr(config, "sequence_min_frames", _DEFAULT_SEQUENCE_MIN_FRAMES),
            default=_DEFAULT_SEQUENCE_MIN_FRAMES,
            minimum=_MIN_SEQUENCE_MIN_FRAMES,
            maximum=_MAX_SEQUENCE_MIN_FRAMES,
        )
        minimum_feasible_window_s = min(
            _MAX_SEQUENCE_WINDOW_S,
            requested_sequence_min_frames / frame_rate,
        )
        sequence_window_s = _coerce_bounded_float(
            _safe_getattr(config, "sequence_window_s", _DEFAULT_SEQUENCE_WINDOW_S),
            default=_DEFAULT_SEQUENCE_WINDOW_S,
            minimum=_MIN_SEQUENCE_WINDOW_S,
            maximum=_MAX_SEQUENCE_WINDOW_S,
        )
        if sequence_window_s < minimum_feasible_window_s:
            sequence_window_s = minimum_feasible_window_s
        max_feasible_frames = _max_sequence_min_frames(
            sequence_window_s=sequence_window_s,
            frame_rate=frame_rate,
        )

        return cls(
            pose_model_path=_safe_getattr(
                config,
                "mediapipe_pose_model_path",
                _DEFAULT_MEDIAPIPE_POSE_MODEL,
            ),
            hand_landmarker_model_path=_safe_getattr(
                config,
                "mediapipe_hand_landmarker_model_path",
                _DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL,
            ),
            gesture_model_path=_safe_getattr(
                config,
                "mediapipe_gesture_model_path",
                _DEFAULT_MEDIAPIPE_GESTURE_MODEL,
            ),
            custom_gesture_model_path=_safe_getattr(
                config,
                "mediapipe_custom_gesture_model_path",
                None,
            ),
            num_hands=_safe_getattr(config, "mediapipe_num_hands", 2),
            attention_score_threshold=_safe_getattr(
                config,
                "attention_score_threshold",
                _DEFAULT_ATTENTION_SCORE_THRESHOLD,
            ),
            sequence_window_s=sequence_window_s,
            sequence_min_frames=min(requested_sequence_min_frames, max_feasible_frames),
            builtin_gesture_min_score=_safe_getattr(config, "builtin_gesture_min_score", 0.35),
            custom_gesture_min_score=_safe_getattr(config, "custom_gesture_min_score", 0.45),
            min_pose_detection_confidence=_safe_getattr(config, "min_pose_detection_confidence", 0.50),
            min_pose_presence_confidence=_safe_getattr(config, "min_pose_presence_confidence", 0.50),
            min_pose_tracking_confidence=_safe_getattr(config, "min_pose_tracking_confidence", 0.50),
            min_hand_detection_confidence=_safe_getattr(config, "min_hand_detection_confidence", 0.35),
            min_hand_presence_confidence=_safe_getattr(config, "min_hand_presence_confidence", 0.35),
            min_hand_tracking_confidence=_safe_getattr(config, "min_hand_tracking_confidence", 0.35),
            max_roi_candidates=_safe_getattr(config, "max_roi_candidates", 4),
            primary_person_roi_padding=_safe_getattr(config, "primary_person_roi_padding", 0.18),
            primary_person_upper_body_ratio=_safe_getattr(config, "primary_person_upper_body_ratio", 0.78),
            wrist_roi_scale=_safe_getattr(config, "wrist_roi_scale", 0.34),
            live_pending_result_timeout_s=_safe_getattr(config, "live_pending_result_timeout_s", 1.0),
            live_gesture_mode=_safe_getattr(
                config,
                "mediapipe_live_gesture_mode",
                _safe_getattr(
                    config,
                    "live_gesture_mode",
                    os.getenv("TWINR_MEDIAPIPE_LIVE_GESTURE_MODE", "video"),
                ),
            ),
            native_heap_trim_interval_s=cast(
                float,
                _safe_getattr(
                    config,
                    "mediapipe_native_heap_trim_interval_s",
                    _safe_getattr(
                        config,
                        "native_heap_trim_interval_s",
                        os.getenv(
                            "TWINR_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S",
                            _DEFAULT_MEDIAPIPE_NATIVE_HEAP_TRIM_INTERVAL_S,
                        ),
                    ),
                ),
            ),
        )


def _safe_getattr(source: object, name: str, default: object) -> object:
    """Get one attribute defensively without letting descriptor errors abort startup."""

    try:
        return getattr(source, name, default)
    except Exception:
        return default


def _resolve_default_custom_gesture_model_path(config: object) -> str | None:
    """Enable the staged custom gesture model automatically when it exists.

    OK-sign support depends on the optional custom gesture recognizer. Twinr
    should use the canonical local asset automatically once the deployment
    already staged it under ``state/mediapipe/models/custom_gesture.task``.
    """

    project_root = Path(str(getattr(config, "project_root", "") or "").strip())
    if not project_root.exists():
        return None
    candidate = project_root / _DEFAULT_MEDIAPIPE_CUSTOM_GESTURE_MODEL
    if not candidate.is_file():
        return None
    return _DEFAULT_MEDIAPIPE_CUSTOM_GESTURE_MODEL


def _coerce_local_path(
    value: object,
    *,
    default: str | None,
    allow_none: bool = False,
    expected_suffixes: tuple[str, ...] | None = None,
) -> str | None:
    """Coerce one value into a sanitized local filesystem path."""

    fallback = None if allow_none else default
    raw = str(value or "").strip()
    if not raw:
        return fallback
    if any(ord(character) < 32 for character in raw):
        return fallback
    lowered = raw.lower()
    # AUDIT-FIX(#2): Reject URLs, Windows-drive paths, backslashes, and traversal segments before any file-open step.
    if "://" in lowered or lowered.startswith(("file:", "http:", "https:")):
        return fallback
    if "\\" in raw or raw.startswith("~") or (len(raw) >= 2 and raw[1] == ":"):
        return fallback

    normalized = str(PurePath(raw))
    if not normalized or normalized == ".":
        return fallback
    if any(part == ".." for part in PurePath(normalized).parts):
        return fallback
    if expected_suffixes is not None and not normalized.lower().endswith(expected_suffixes):
        return fallback
    return normalized


def _coerce_main_size(value: object, *, default: tuple[int, int]) -> tuple[int, int]:
    """Coerce one main camera size into a bounded ``(width, height)`` tuple."""

    if isinstance(value, (list, tuple)) and len(value) == 2:
        raw_width, raw_height = value
    else:
        raw_width, raw_height = default
    width = _coerce_bounded_int(
        raw_width,
        default=default[0],
        minimum=_MIN_CAMERA_WIDTH,
        maximum=_MAX_CAMERA_WIDTH,
    )
    height = _coerce_bounded_int(
        raw_height,
        default=default[1],
        minimum=_MIN_CAMERA_HEIGHT,
        maximum=_MAX_CAMERA_HEIGHT,
    )
    return (width, height)


def _max_sequence_min_frames(*, sequence_window_s: float, frame_rate: int) -> int:
    """Compute one feasible upper bound for sequence frames from window and FPS."""

    feasible_frames = int(math.floor(sequence_window_s * frame_rate))
    return max(_MIN_SEQUENCE_MIN_FRAMES, min(feasible_frames, _MAX_SEQUENCE_MIN_FRAMES))


def _coerce_choice(value: object, *, default: str, allowed: frozenset[str]) -> str:
    """Coerce one string value into an allowed choice."""

    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        return default
    return normalized


def _coerce_float(value: object, *, default: float) -> float:
    """Coerce one value to a finite float."""

    # AUDIT-FIX(#5): Reject bools so accidental True/False config values do not become 1.0/0.0.
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _coerce_int(value: object, *, default: int) -> int:
    """Coerce one value to an integer."""

    # AUDIT-FIX(#5): Reject bools so accidental True/False config values do not become 1/0.
    if isinstance(value, bool):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number


def _coerce_positive_int(value: object, *, default: int) -> int:
    """Coerce one value to a positive integer."""

    number = _coerce_int(value, default=default)
    if number <= 0:
        return default
    return number


def _coerce_bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    """Coerce one value to an integer within inclusive bounds."""

    number = _coerce_int(value, default=default)
    if number < minimum or number > maximum:
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


def _coerce_bounded_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Coerce one value to a finite float within inclusive bounds."""

    number = _coerce_float(value, default=default)
    if number < minimum or number > maximum:
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


__all__ = [
    "AICameraAdapterConfig",
    "DEFAULT_MEDIAPIPE_GESTURE_MODEL_URL",
    "DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL",
    "DEFAULT_MEDIAPIPE_POSE_MODEL_URL",
    "MediaPipeVisionConfig",
    "_clamp_ratio",
    "_coerce_float",
    "_coerce_non_negative_float",
    "_coerce_positive_float",
    "_coerce_positive_int",
]
