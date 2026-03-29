# CHANGELOG: 2026-03-28
# BUG-1: De-duplicated same-hand detections across overlapping ROIs so one hand can no longer occupy both output slots and suppress the second hand.
# BUG-2: Replaced unstable handedness/large-ROI result ordering with occupancy-aware ranking that prefers tighter, better-localized hand crops.
# SEC-1: Serialized access to the native MediaPipe task so shared-worker concurrent calls cannot race detect()/close() and crash the process.
# IMP-1: Added temporal full-frame hand ROI tracking to recover hands when pose wrists flicker and to emulate part of MediaPipe VIDEO-mode tracking while keeping crop-local IMAGE-mode inference.
# IMP-2: Exposed MediaPipe world landmarks and upgraded wrist ROI generation with forward-biased arm-vector crops.

"""Resolve bounded hand-landmark ROIs for the Pi-side MediaPipe camera path.

This module owns the local hand ROI selection and MediaPipe Hand Landmarker
execution that complement Twinr's IMX500 person gate. It keeps the ROI logic
bounded, maps crop-local landmarks back into full-frame normalized coordinates,
and returns typed detections that higher layers can reuse for gesture
recognition without re-implementing crop math.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, cast
import math
import threading

from twinr.agent.workflows.forensics import workflow_decision, workflow_span
from twinr.hardware.camera_ai.mediapipe_runtime import normalize_image_data
from twinr.hardware.camera_ai.models import AICameraBox


DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_WRIST_SCORE_THRESHOLD = 0.35
_ELBOW_SCORE_THRESHOLD = 0.30
_SHOULDER_SCORE_THRESHOLD = 0.30

_LOCAL_HAND_CROP_PADDING = 0.14
_LOCAL_HAND_MIN_CONTEXT_RATIO = 0.30
_LOCAL_HAND_RETRY_CROP_PADDING = 0.24
_LOCAL_HAND_RETRY_MIN_CONTEXT_RATIO = 0.42

_WIDE_CONTEXT_HORIZONTAL_PADDING = 0.45
_WIDE_CONTEXT_TOP_PADDING = 0.12
_WIDE_CONTEXT_BOTTOM_PADDING = 0.55

_HAND_BOX_PADDING = 0.08
_HAND_BOX_MIN_CONTEXT_RATIO = 0.16
_DEFAULT_MIN_UNIQUE_HAND_IOU = 0.42
_DEFAULT_TEMPORAL_ROI_PADDING = 0.20
_DEFAULT_TEMPORAL_ROI_TTL_MS = 900
_DEFAULT_TEMPORAL_ROI_MAX_CANDIDATES = 2
_DEFAULT_WRIST_ROI_CENTER_FORWARD_BIAS = 0.28

_FULL_FRAME_PRIORITY = 99
_TEMPORAL_TRACK_PRIORITY = 0
_WRIST_PRIORITY = 1
_UPPER_BODY_PRIORITY = 2
_FULL_BODY_PRIORITY = 3

_SOURCE_PREFERENCE = {
    "temporal": 1.00,
    "wrist": 0.90,
    "upper_body": 0.60,
    "full_body": 0.45,
    "wide_wrist": 0.40,
    "wide_upper_body": 0.30,
    "wide_full_body": 0.20,
    "full_frame": 0.10,
}


class HandRoiSource(StrEnum):
    """Describe how one hand ROI candidate was derived."""

    FULL_FRAME = "full_frame"
    PRIMARY_PERSON_FULL_BODY = "primary_person_full_body"
    PRIMARY_PERSON_UPPER_BODY = "primary_person_upper_body"
    PRIMARY_PERSON_WIDE_FULL_BODY = "primary_person_wide_full_body"
    PRIMARY_PERSON_WIDE_UPPER_BODY = "primary_person_wide_upper_body"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    WIDE_LEFT_WRIST = "wide_left_wrist"
    WIDE_RIGHT_WRIST = "wide_right_wrist"
    TEMPORAL_LEFT = "temporal_left"
    TEMPORAL_RIGHT = "temporal_right"
    TEMPORAL_UNKNOWN = "temporal_unknown"


@dataclass(frozen=True, slots=True)
class HandLandmarkWorkerConfig:
    """Store bounded runtime settings for the local hand-landmark worker."""

    model_path: str
    num_hands: int = 2
    max_roi_candidates: int = 4
    primary_person_roi_padding: float = 0.18
    primary_person_upper_body_ratio: float = 0.78
    wrist_roi_scale: float = 0.34
    wrist_roi_center_forward_bias: float = _DEFAULT_WRIST_ROI_CENTER_FORWARD_BIAS
    wide_context_horizontal_padding: float = _WIDE_CONTEXT_HORIZONTAL_PADDING
    wide_context_top_padding: float = _WIDE_CONTEXT_TOP_PADDING
    wide_context_bottom_padding: float = _WIDE_CONTEXT_BOTTOM_PADDING
    min_unique_hand_iou: float = _DEFAULT_MIN_UNIQUE_HAND_IOU
    enable_temporal_roi_tracking: bool = True
    temporal_roi_padding: float = _DEFAULT_TEMPORAL_ROI_PADDING
    temporal_roi_ttl_ms: int = _DEFAULT_TEMPORAL_ROI_TTL_MS
    temporal_roi_max_candidates: int = _DEFAULT_TEMPORAL_ROI_MAX_CANDIDATES
    min_hand_detection_confidence: float = 0.35
    min_hand_presence_confidence: float = 0.35
    min_hand_tracking_confidence: float = 0.35

    @classmethod
    def from_config(cls, config: object) -> "HandLandmarkWorkerConfig":
        """Build one worker config from a camera-config-like object."""

        return cls(
            model_path=str(getattr(config, "hand_landmarker_model_path", "") or "").strip(),
            num_hands=_coerce_int(getattr(config, "num_hands", 2), default=2, minimum=1),
            max_roi_candidates=_coerce_int(
                getattr(config, "max_roi_candidates", 4),
                default=4,
                minimum=1,
            ),
            primary_person_roi_padding=_clamp_ratio(
                getattr(config, "primary_person_roi_padding", 0.18),
                default=0.18,
            ),
            primary_person_upper_body_ratio=_clamp_ratio(
                getattr(config, "primary_person_upper_body_ratio", 0.78),
                default=0.78,
            ),
            wrist_roi_scale=_clamp_ratio(
                getattr(config, "wrist_roi_scale", 0.34),
                default=0.34,
            ),
            wrist_roi_center_forward_bias=_clamp_ratio(
                getattr(config, "wrist_roi_center_forward_bias", _DEFAULT_WRIST_ROI_CENTER_FORWARD_BIAS),
                default=_DEFAULT_WRIST_ROI_CENTER_FORWARD_BIAS,
            ),
            wide_context_horizontal_padding=_clamp_ratio(
                getattr(config, "wide_context_horizontal_padding", _WIDE_CONTEXT_HORIZONTAL_PADDING),
                default=_WIDE_CONTEXT_HORIZONTAL_PADDING,
            ),
            wide_context_top_padding=_clamp_ratio(
                getattr(config, "wide_context_top_padding", _WIDE_CONTEXT_TOP_PADDING),
                default=_WIDE_CONTEXT_TOP_PADDING,
            ),
            wide_context_bottom_padding=_clamp_ratio(
                getattr(config, "wide_context_bottom_padding", _WIDE_CONTEXT_BOTTOM_PADDING),
                default=_WIDE_CONTEXT_BOTTOM_PADDING,
            ),
            min_unique_hand_iou=_clamp_ratio(
                getattr(config, "min_unique_hand_iou", _DEFAULT_MIN_UNIQUE_HAND_IOU),
                default=_DEFAULT_MIN_UNIQUE_HAND_IOU,
            ),
            enable_temporal_roi_tracking=_coerce_bool(
                getattr(config, "enable_temporal_roi_tracking", True),
                default=True,
            ),
            temporal_roi_padding=_clamp_ratio(
                getattr(config, "temporal_roi_padding", _DEFAULT_TEMPORAL_ROI_PADDING),
                default=_DEFAULT_TEMPORAL_ROI_PADDING,
            ),
            temporal_roi_ttl_ms=_coerce_int(
                getattr(config, "temporal_roi_ttl_ms", _DEFAULT_TEMPORAL_ROI_TTL_MS),
                default=_DEFAULT_TEMPORAL_ROI_TTL_MS,
                minimum=0,
            ),
            temporal_roi_max_candidates=_coerce_int(
                getattr(config, "temporal_roi_max_candidates", _DEFAULT_TEMPORAL_ROI_MAX_CANDIDATES),
                default=_DEFAULT_TEMPORAL_ROI_MAX_CANDIDATES,
                minimum=0,
            ),
            min_hand_detection_confidence=_clamp_ratio(
                getattr(config, "min_hand_detection_confidence", 0.35),
                default=0.35,
            ),
            min_hand_presence_confidence=_clamp_ratio(
                getattr(config, "min_hand_presence_confidence", 0.35),
                default=0.35,
            ),
            min_hand_tracking_confidence=_clamp_ratio(
                getattr(config, "min_hand_tracking_confidence", 0.35),
                default=0.35,
            ),
        )


@dataclass(frozen=True, slots=True)
class HandLandmarkPoint:
    """Describe one full-frame normalized hand landmark."""

    x: float
    y: float
    z: float
    score: float | None = None


@dataclass(frozen=True, slots=True)
class HandWorldLandmarkPoint:
    """Describe one hand landmark in MediaPipe world coordinates (meters)."""

    x: float
    y: float
    z: float
    score: float | None = None


@dataclass(frozen=True, slots=True)
class HandLandmarkDetection:
    """Describe one hand-landmark detection mapped back to full-frame space."""

    roi: AICameraBox
    roi_source: HandRoiSource
    handedness: str | None
    handedness_score: float | None
    landmarks: tuple[HandLandmarkPoint, ...]
    roi_frame_rgb: Any
    gesture_frame_rgb: Any | None = None
    gesture_context_frame_rgb: Any | None = None
    world_landmarks: tuple[HandWorldLandmarkPoint, ...] = ()
    hand_box: AICameraBox | None = None
    roi_occupancy: float | None = None
    roi_priority: int = 0

    @property
    def confidence(self) -> float:
        """Return the best available detection confidence proxy."""

        if self.handedness_score is not None:
            return self.handedness_score
        if not self.landmarks:
            return 0.0
        scores = [point.score for point in self.landmarks if point.score is not None]
        if not scores:
            return 0.0
        return round(sum(scores) / float(len(scores)), 3)


@dataclass(frozen=True, slots=True)
class HandLandmarkResult:
    """Describe the bounded set of hand ROI detections for one frame."""

    detections: tuple[HandLandmarkDetection, ...] = ()
    final_timestamp_ms: int | None = None

    @property
    def primary_detection(self) -> HandLandmarkDetection | None:
        """Return the highest-quality detection when available."""

        if not self.detections:
            return None
        return self.detections[0]


@dataclass(frozen=True, slots=True)
class _HandRoiCandidate:
    """Describe one bounded hand ROI candidate before inference."""

    box: AICameraBox
    source: HandRoiSource
    priority: int


@dataclass(slots=True)
class _TrackedHand:
    """Store one recent hand track in full-frame normalized coordinates."""

    hand_box: AICameraBox
    handedness: str | None
    last_seen_timestamp_ms: int
    quality_score: float
    velocity_x_per_ms: float = 0.0
    velocity_y_per_ms: float = 0.0


class MediaPipeHandLandmarkWorker:
    """Run MediaPipe Hand Landmarker on bounded candidate ROIs.

    The worker evaluates independent ROI crops derived from the primary person
    box, wrist hints, and recent successful hand detections. These crops are
    not a single stable MediaPipe stream, so the hand landmarker still runs in
    IMAGE mode. A lightweight external temporal ROI cache recovers some of the
    robustness and latency benefit of VIDEO-mode tracking without feeding
    incompatible crop coordinate spaces into the native tracker.
    """

    def __init__(self, *, config: HandLandmarkWorkerConfig) -> None:
        """Initialize one lazy hand-landmark worker."""

        self.config = config
        self._hand_landmarker: Any | None = None
        self._lock = threading.RLock()
        self._tracked_hands: list[_TrackedHand] = []

    def close(self) -> None:
        """Close the active MediaPipe task when supported."""

        with self._lock:
            if self._hand_landmarker is None:
                self._tracked_hands.clear()
                return
            close_fn = getattr(self._hand_landmarker, "close", None)
            if callable(close_fn):
                close_callable = cast(Callable[[], None], close_fn)
                close_callable()  # pylint: disable=not-callable
            self._hand_landmarker = None
            self._tracked_hands.clear()

    def analyze(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
        primary_person_box: AICameraBox,
        sparse_keypoints: dict[int, tuple[float, float, float]] | None = None,
    ) -> HandLandmarkResult:
        """Run hand-landmark inference on bounded ROIs for one frame.

        Each ROI is an independent crop. Use one image-mode inference per crop
        instead of feeding alternating ROI spaces into one video tracker.
        """

        with self._lock:
            with workflow_span(
                name="hand_landmark_worker_analyze",
                kind="io",
                details={"timestamp_ms": timestamp_ms},
            ):
                sparse_keypoints = sparse_keypoints or {}
                self._prune_tracked_hands(timestamp_ms)
                pose_candidates = _build_hand_roi_candidates(
                    primary_person_box=primary_person_box,
                    sparse_keypoints=sparse_keypoints,
                    config=self.config,
                )
                candidates = self._augment_with_temporal_candidates(
                    pose_candidates=pose_candidates,
                    timestamp_ms=timestamp_ms,
                )
                if not candidates:
                    workflow_decision(
                        msg="hand_landmark_roi_strategy",
                        question="Should the hand landmark worker run any person-conditioned ROI pass?",
                        selected={"id": "no_candidates", "summary": "No ROI candidates were available for this frame."},
                        options=[
                            {"id": "primary_candidates", "summary": "Run the focused person-conditioned ROI pass."},
                            {"id": "no_candidates", "summary": "Return without running hand landmark inference."},
                        ],
                        context={"candidate_count": 0},
                        confidence="forensic",
                        guardrails=["hand_roi_candidates_required"],
                        kpi_impact_estimate={"latency": "low"},
                    )
                    return HandLandmarkResult()

                primary_result = self._analyze_candidates(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    candidates=candidates,
                )
                if primary_result.detections:
                    workflow_decision(
                        msg="hand_landmark_roi_strategy",
                        question="Which ROI pass should the hand landmark worker use for this frame?",
                        selected={"id": "primary_candidates", "summary": "The focused person-conditioned ROI pass already found hands."},
                        options=[
                            {"id": "primary_candidates", "summary": "Keep the focused person-conditioned ROI detections."},
                            {"id": "wide_context_rescue", "summary": "Retry with a wider torso/table context pass."},
                        ],
                        context={
                            "primary_candidate_count": len(candidates),
                            "primary_detection_count": len(primary_result.detections),
                        },
                        confidence="forensic",
                        guardrails=["hand_roi_primary_first"],
                        kpi_impact_estimate={"latency": "low", "gesture_accuracy": "primary"},
                    )
                    self._update_tracked_hands(
                        detections=primary_result.detections,
                        timestamp_ms=primary_result.final_timestamp_ms or timestamp_ms,
                    )
                    return primary_result

                rescue_candidates = _build_wide_context_hand_roi_candidates(
                    primary_person_box=primary_person_box,
                    sparse_keypoints=sparse_keypoints,
                    config=self.config,
                )
                if not rescue_candidates:
                    workflow_decision(
                        msg="hand_landmark_roi_strategy",
                        question="Should the hand landmark worker retry with a wide-context ROI pass?",
                        selected={"id": "primary_only", "summary": "No wide-context rescue candidates were available."},
                        options=[
                            {"id": "primary_only", "summary": "Keep only the focused person-conditioned pass."},
                            {"id": "wide_context_rescue", "summary": "Retry with a wider torso/table context pass."},
                        ],
                        context={
                            "primary_candidate_count": len(candidates),
                            "primary_detection_count": len(primary_result.detections),
                            "wide_candidate_count": 0,
                        },
                        confidence="forensic",
                        guardrails=["hand_roi_wide_context_optional"],
                        kpi_impact_estimate={"latency": "low"},
                    )
                    return primary_result

                rescue_timestamp_ms = (
                    primary_result.final_timestamp_ms + 1
                    if primary_result.final_timestamp_ms is not None
                    else timestamp_ms + len(candidates)
                )
                rescue_result = self._analyze_candidates(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=rescue_timestamp_ms,
                    candidates=rescue_candidates,
                )
                workflow_decision(
                    msg="hand_landmark_roi_strategy",
                    question="Should the hand landmark worker promote the wide-context rescue pass?",
                    selected={
                        "id": "wide_context_rescue" if rescue_result.detections else "primary_only",
                        "summary": (
                            "Promote the wide-context rescue detections."
                            if rescue_result.detections
                            else "Keep the empty primary result because the rescue pass also found no hands."
                        ),
                    },
                    options=[
                        {"id": "primary_only", "summary": "Keep only the focused person-conditioned pass."},
                        {"id": "wide_context_rescue", "summary": "Promote the wider torso/table context pass."},
                    ],
                    context={
                        "primary_candidate_count": len(candidates),
                        "primary_detection_count": len(primary_result.detections),
                        "wide_candidate_count": len(rescue_candidates),
                        "wide_detection_count": len(rescue_result.detections),
                    },
                    confidence="forensic",
                    guardrails=["hand_roi_wide_context_optional"],
                    kpi_impact_estimate={"latency": "medium", "gesture_accuracy": "rescue"},
                )
                if rescue_result.detections:
                    self._update_tracked_hands(
                        detections=rescue_result.detections,
                        timestamp_ms=rescue_result.final_timestamp_ms or rescue_timestamp_ms,
                    )
                    return rescue_result
                return primary_result

    def analyze_full_frame(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
    ) -> HandLandmarkResult:
        """Run one bounded whole-frame hand-landmark pass.

        The gesture fast path normally prefers person-conditioned ROIs, but a
        visible hand can still matter when IMX500 loses the person box or only
        a hand is visible above an occluder. Keep this rescue explicit and
        bounded to one full-frame IMAGE-mode pass.
        """

        with self._lock:
            result = self._analyze_candidates(
                runtime=runtime,
                frame_rgb=frame_rgb,
                timestamp_ms=timestamp_ms,
                candidates=(
                    _HandRoiCandidate(
                        box=AICameraBox(top=0.0, left=0.0, bottom=1.0, right=1.0),
                        source=HandRoiSource.FULL_FRAME,
                        priority=_FULL_FRAME_PRIORITY,
                    ),
                ),
            )
            if result.detections:
                self._update_tracked_hands(
                    detections=result.detections,
                    timestamp_ms=result.final_timestamp_ms or timestamp_ms,
                )
            return result

    def _augment_with_temporal_candidates(
        self,
        *,
        pose_candidates: tuple[_HandRoiCandidate, ...],
        timestamp_ms: int,
    ) -> tuple[_HandRoiCandidate, ...]:
        """Inject recent hand tracks when current pose wrists are incomplete."""

        if not self.config.enable_temporal_roi_tracking:
            return pose_candidates
        if self.config.temporal_roi_max_candidates <= 0:
            return pose_candidates
        if not self._tracked_hands:
            return pose_candidates

        wrist_candidate_count = sum(1 for candidate in pose_candidates if _is_wrist_source(candidate.source))
        if pose_candidates and wrist_candidate_count >= self.config.num_hands:
            return pose_candidates

        temporal_candidates = _build_temporal_hand_roi_candidates(
            tracked_hands=tuple(self._tracked_hands),
            timestamp_ms=timestamp_ms,
            config=self.config,
        )
        if not temporal_candidates:
            return pose_candidates
        return _merge_hand_roi_candidates(
            temporal_candidates,
            pose_candidates,
            limit=self.config.max_roi_candidates,
        )

    def _analyze_candidates(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
        candidates: tuple[_HandRoiCandidate, ...],
    ) -> HandLandmarkResult:
        """Run one IMAGE-mode hand-landmarker pass across explicit ROI candidates."""

        hand_landmarker = self._ensure_hand_landmarker(runtime)
        mp_runtime = _require_runtime_item(runtime, "mp")
        detections: list[HandLandmarkDetection] = []
        final_timestamp_ms = None
        for index, candidate in enumerate(candidates):
            crop = _crop_frame_rgb(frame_rgb, candidate.box)
            if crop is None:
                continue
            image_data = normalize_image_data(crop)
            image = mp_runtime.Image(image_format=mp_runtime.ImageFormat.SRGB, data=image_data)
            candidate_timestamp_ms = timestamp_ms + index
            result = hand_landmarker.detect(image)
            final_timestamp_ms = candidate_timestamp_ms
            parsed = _parse_hand_landmark_result(
                result=result,
                roi=candidate.box,
                roi_source=candidate.source,
                roi_priority=candidate.priority,
                roi_frame_rgb=image_data,
                full_frame_rgb=frame_rgb,
            )
            detections = _merge_unique_hand_detections(
                existing=detections,
                incoming=parsed,
                min_iou=self.config.min_unique_hand_iou,
            )
        detections.sort(key=_detection_sort_key, reverse=True)
        return HandLandmarkResult(
            detections=tuple(detections[: self.config.num_hands]),
            final_timestamp_ms=final_timestamp_ms,
        )

    def _ensure_hand_landmarker(self, runtime: dict[str, Any]) -> Any:
        """Reuse or create the configured hand landmarker.

        ROI crops are independent images, so IMAGE mode is the correct
        MediaPipe running mode for this worker.
        """

        if self._hand_landmarker is not None:
            return self._hand_landmarker
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"mediapipe_hand_landmarker_model_missing:{model_path}")

        vision = _require_runtime_item(runtime, "vision")
        base_options = _require_runtime_item(runtime, "BaseOptions")
        options = vision.HandLandmarkerOptions(
            base_options=base_options(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_hand_tracking_confidence,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)
        return self._hand_landmarker

    def _prune_tracked_hands(self, timestamp_ms: int) -> None:
        """Drop stale temporal hand tracks."""

        ttl_ms = self.config.temporal_roi_ttl_ms
        if ttl_ms <= 0:
            self._tracked_hands.clear()
            return
        self._tracked_hands = [
            track
            for track in self._tracked_hands
            if timestamp_ms - track.last_seen_timestamp_ms <= ttl_ms
        ]

    def _update_tracked_hands(
        self,
        *,
        detections: tuple[HandLandmarkDetection, ...],
        timestamp_ms: int,
    ) -> None:
        """Refresh temporal hand tracks from the latest accepted detections."""

        if not detections:
            self._prune_tracked_hands(timestamp_ms)
            return

        self._prune_tracked_hands(timestamp_ms)
        updated_tracks = list(self._tracked_hands)
        for detection in sorted(detections, key=_detection_sort_key, reverse=True):
            if detection.hand_box is None:
                continue
            match_index = _find_best_track_match(updated_tracks, detection)
            detection_center_x, detection_center_y = _box_center(detection.hand_box)
            quality_score = _detection_rank_scalar(detection)

            if match_index is None:
                updated_tracks.append(
                    _TrackedHand(
                        hand_box=detection.hand_box,
                        handedness=detection.handedness,
                        last_seen_timestamp_ms=timestamp_ms,
                        quality_score=quality_score,
                    )
                )
                continue

            previous = updated_tracks[match_index]
            previous_center_x, previous_center_y = _box_center(previous.hand_box)
            delta_ms = max(1, timestamp_ms - previous.last_seen_timestamp_ms)
            updated_tracks[match_index] = _TrackedHand(
                hand_box=detection.hand_box,
                handedness=detection.handedness or previous.handedness,
                last_seen_timestamp_ms=timestamp_ms,
                quality_score=max(previous.quality_score * 0.75, quality_score),
                velocity_x_per_ms=(detection_center_x - previous_center_x) / float(delta_ms),
                velocity_y_per_ms=(detection_center_y - previous_center_y) / float(delta_ms),
            )

        updated_tracks.sort(
            key=lambda track: (track.last_seen_timestamp_ms, track.quality_score),
            reverse=True,
        )
        self._tracked_hands = updated_tracks[: self.config.num_hands]


def _build_hand_roi_candidates(
    *,
    primary_person_box: AICameraBox,
    sparse_keypoints: dict[int, tuple[float, float, float]],
    config: HandLandmarkWorkerConfig,
) -> tuple[_HandRoiCandidate, ...]:
    """Return bounded ROI candidates for one primary person.

    Weak or stale pose wrists are common on the Pi when the user keeps a hand
    close to the torso. Keep the focused wrist crops first for efficiency, but
    always retain the broader person rescue crops as bounded second chances so
    a noisy wrist hint does not disable the only ROIs that still contain the
    real hand.
    """

    candidates: list[_HandRoiCandidate] = []
    left_wrist_candidate = _build_wrist_roi_candidate(
        wrist_key=9,
        elbow_key=7,
        shoulder_key=5,
        primary_person_box=primary_person_box,
        sparse_keypoints=sparse_keypoints,
        config=config,
        source=HandRoiSource.LEFT_WRIST,
    )
    if left_wrist_candidate is not None:
        candidates.append(left_wrist_candidate)
    right_wrist_candidate = _build_wrist_roi_candidate(
        wrist_key=10,
        elbow_key=8,
        shoulder_key=6,
        primary_person_box=primary_person_box,
        sparse_keypoints=sparse_keypoints,
        config=config,
        source=HandRoiSource.RIGHT_WRIST,
    )
    if right_wrist_candidate is not None:
        candidates.append(right_wrist_candidate)
    candidates.append(
        _HandRoiCandidate(
            box=_build_primary_person_upper_body_roi(
                primary_person_box=primary_person_box,
                config=config,
            ),
            source=HandRoiSource.PRIMARY_PERSON_UPPER_BODY,
            priority=_UPPER_BODY_PRIORITY,
        )
    )
    candidates.append(
        _HandRoiCandidate(
            box=_build_primary_person_full_body_roi(
                primary_person_box=primary_person_box,
                config=config,
            ),
            source=HandRoiSource.PRIMARY_PERSON_FULL_BODY,
            priority=_FULL_BODY_PRIORITY,
        )
    )
    return _dedupe_and_limit_candidates(
        candidates=candidates,
        limit=config.max_roi_candidates,
    )


def _build_wide_context_hand_roi_candidates(
    *,
    primary_person_box: AICameraBox,
    sparse_keypoints: dict[int, tuple[float, float, float]],
    config: HandLandmarkWorkerConfig,
) -> tuple[_HandRoiCandidate, ...]:
    """Return a second bounded ROI set with wider torso/table context.

    Pi repro frames showed that the default person-conditioned crops can miss
    hands entirely in seated table scenes even when a person box is present.
    Keep this pass bounded and ordered after the focused primary pass.
    """

    rescue_context_box = _build_primary_person_wide_context_box(
        primary_person_box=primary_person_box,
        config=config,
    )
    if rescue_context_box == primary_person_box:
        return ()

    candidates: list[_HandRoiCandidate] = []
    left_wrist_candidate = _build_wrist_roi_candidate(
        wrist_key=9,
        elbow_key=7,
        shoulder_key=5,
        primary_person_box=rescue_context_box,
        sparse_keypoints=sparse_keypoints,
        config=config,
        source=HandRoiSource.WIDE_LEFT_WRIST,
    )
    if left_wrist_candidate is not None:
        candidates.append(left_wrist_candidate)
    right_wrist_candidate = _build_wrist_roi_candidate(
        wrist_key=10,
        elbow_key=8,
        shoulder_key=6,
        primary_person_box=rescue_context_box,
        sparse_keypoints=sparse_keypoints,
        config=config,
        source=HandRoiSource.WIDE_RIGHT_WRIST,
    )
    if right_wrist_candidate is not None:
        candidates.append(right_wrist_candidate)
    candidates.append(
        _HandRoiCandidate(
            box=_build_primary_person_upper_body_roi(
                primary_person_box=rescue_context_box,
                config=config,
                padding_override=0.0,
            ),
            source=HandRoiSource.PRIMARY_PERSON_WIDE_UPPER_BODY,
            priority=_UPPER_BODY_PRIORITY,
        )
    )
    candidates.append(
        _HandRoiCandidate(
            box=_build_primary_person_full_body_roi(
                primary_person_box=rescue_context_box,
                config=config,
                padding_override=0.0,
            ),
            source=HandRoiSource.PRIMARY_PERSON_WIDE_FULL_BODY,
            priority=_FULL_BODY_PRIORITY,
        )
    )
    return _dedupe_and_limit_candidates(
        candidates=candidates,
        limit=config.max_roi_candidates,
    )


def _build_temporal_hand_roi_candidates(
    *,
    tracked_hands: tuple[_TrackedHand, ...],
    timestamp_ms: int,
    config: HandLandmarkWorkerConfig,
) -> tuple[_HandRoiCandidate, ...]:
    """Return candidates derived from recent successful hand detections."""

    candidates: list[_HandRoiCandidate] = []
    for tracked_hand in tracked_hands:
        age_ms = max(0, timestamp_ms - tracked_hand.last_seen_timestamp_ms)
        if age_ms > config.temporal_roi_ttl_ms:
            continue
        predicted_box = _predict_temporal_roi_box(
            tracked_hand=tracked_hand,
            age_ms=age_ms,
            config=config,
        )
        if predicted_box is None:
            continue
        candidates.append(
            _HandRoiCandidate(
                box=predicted_box,
                source=_temporal_source_for_handedness(tracked_hand.handedness),
                priority=_TEMPORAL_TRACK_PRIORITY,
            )
        )
    return _dedupe_and_limit_candidates(
        candidates=candidates,
        limit=config.temporal_roi_max_candidates,
    )


def _merge_hand_roi_candidates(
    *candidate_groups: tuple[_HandRoiCandidate, ...],
    limit: int,
) -> tuple[_HandRoiCandidate, ...]:
    """Merge candidate groups while preserving priority and bounded size."""

    merged: list[_HandRoiCandidate] = []
    for group in candidate_groups:
        merged.extend(group)
    return _dedupe_and_limit_candidates(candidates=merged, limit=limit)


def _dedupe_and_limit_candidates(
    *,
    candidates: list[_HandRoiCandidate],
    limit: int,
) -> tuple[_HandRoiCandidate, ...]:
    """Sort, dedupe, and bound one candidate list."""

    if limit <= 0:
        return ()
    candidates.sort(key=lambda item: (item.priority, item.box.area))
    deduped: list[_HandRoiCandidate] = []
    for candidate in candidates:
        if any(_should_dedupe_candidate(candidate, existing) for existing in deduped):
            continue
        deduped.append(candidate)
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def _build_primary_person_upper_body_roi(
    *,
    primary_person_box: AICameraBox,
    config: HandLandmarkWorkerConfig,
    padding_override: float | None = None,
) -> AICameraBox:
    """Expand the primary person box into one upper-body hand-search ROI."""

    padding = config.primary_person_roi_padding if padding_override is None else padding_override
    horizontal_padding = primary_person_box.width * padding
    vertical_padding = primary_person_box.height * padding
    return AICameraBox(
        top=primary_person_box.top - vertical_padding,
        left=primary_person_box.left - horizontal_padding,
        bottom=(
            primary_person_box.top
            + (primary_person_box.height * config.primary_person_upper_body_ratio)
            + vertical_padding
        ),
        right=primary_person_box.right + horizontal_padding,
    )


def _build_primary_person_full_body_roi(
    *,
    primary_person_box: AICameraBox,
    config: HandLandmarkWorkerConfig,
    padding_override: float | None = None,
) -> AICameraBox:
    """Expand the primary person box into one full-body rescue ROI."""

    padding = config.primary_person_roi_padding if padding_override is None else padding_override
    horizontal_padding = primary_person_box.width * padding
    vertical_padding = primary_person_box.height * padding
    return AICameraBox(
        top=primary_person_box.top - vertical_padding,
        left=primary_person_box.left - horizontal_padding,
        bottom=primary_person_box.bottom + vertical_padding,
        right=primary_person_box.right + horizontal_padding,
    )


def _build_primary_person_wide_context_box(
    *,
    primary_person_box: AICameraBox,
    config: HandLandmarkWorkerConfig,
) -> AICameraBox:
    """Return one wider bounded person context for seated/table rescues."""

    horizontal_padding = primary_person_box.width * config.wide_context_horizontal_padding
    top_padding = primary_person_box.height * config.wide_context_top_padding
    bottom_padding = primary_person_box.height * config.wide_context_bottom_padding
    return AICameraBox(
        top=primary_person_box.top - top_padding,
        left=primary_person_box.left - horizontal_padding,
        bottom=primary_person_box.bottom + bottom_padding,
        right=primary_person_box.right + horizontal_padding,
    )


def _build_wrist_roi_candidate(
    *,
    wrist_key: int,
    elbow_key: int,
    shoulder_key: int,
    primary_person_box: AICameraBox,
    sparse_keypoints: dict[int, tuple[float, float, float]],
    config: HandLandmarkWorkerConfig,
    source: HandRoiSource,
) -> _HandRoiCandidate | None:
    """Build one wrist-centered ROI candidate when the pose is usable.

    The legacy crop centered exactly on the wrist, which often wastes half the
    crop on forearm pixels and clips extended fingers. Shift the square crop
    forward along the arm direction when elbow/shoulder evidence is available.
    """

    wrist = sparse_keypoints.get(wrist_key)
    if wrist is None or wrist[2] < _WRIST_SCORE_THRESHOLD:
        return None

    elbow = sparse_keypoints.get(elbow_key)
    shoulder = sparse_keypoints.get(shoulder_key)
    size = max(
        primary_person_box.width * config.wrist_roi_scale,
        primary_person_box.height * config.wrist_roi_scale * 0.7,
        0.14,
    )
    if elbow is not None and elbow[2] >= _ELBOW_SCORE_THRESHOLD:
        size = max(size, _distance_2d(wrist, elbow) * 2.4)
    if shoulder is not None and shoulder[2] >= _SHOULDER_SCORE_THRESHOLD:
        size = max(size, _distance_2d(wrist, shoulder) * 1.15)
    size = min(0.48, size)

    center_x = wrist[0]
    center_y = wrist[1]
    direction_x = 0.0
    direction_y = 0.0
    if elbow is not None and elbow[2] >= _ELBOW_SCORE_THRESHOLD:
        direction_x, direction_y = _normalize_vector_2d(wrist[0] - elbow[0], wrist[1] - elbow[1])
    elif shoulder is not None and shoulder[2] >= _SHOULDER_SCORE_THRESHOLD:
        direction_x, direction_y = _normalize_vector_2d(wrist[0] - shoulder[0], wrist[1] - shoulder[1])

    center_x += direction_x * size * config.wrist_roi_center_forward_bias
    center_y += direction_y * size * config.wrist_roi_center_forward_bias
    half = size / 2.0
    return _HandRoiCandidate(
        box=AICameraBox(
            top=center_y - half,
            left=center_x - half,
            bottom=center_y + half,
            right=center_x + half,
        ),
        source=source,
        priority=_WRIST_PRIORITY,
    )


def _parse_hand_landmark_result(
    *,
    result: Any,
    roi: AICameraBox,
    roi_source: HandRoiSource,
    roi_priority: int,
    roi_frame_rgb: Any,
    full_frame_rgb: Any | None = None,
) -> tuple[HandLandmarkDetection, ...]:
    """Map one MediaPipe hand-landmarker result into typed detections."""

    hand_landmarks = (
        getattr(result, "hand_landmarks", None)
        or getattr(result, "landmarks", None)
        or ()
    )
    hand_world_landmarks = (
        getattr(result, "hand_world_landmarks", None)
        or getattr(result, "world_landmarks", None)
        or getattr(result, "worldLandmarks", None)
        or ()
    )
    handedness = getattr(result, "handedness", None) or getattr(result, "handednesses", None) or ()

    detections: list[HandLandmarkDetection] = []
    for index, local_landmarks in enumerate(hand_landmarks):
        handedness_label, handedness_score = _resolve_handedness(handedness, index=index)
        projected_landmarks = tuple(
            _project_landmark_to_full_frame(
                landmark=landmark,
                roi=roi,
            )
            for landmark in local_landmarks or ()
        )
        projected_world_landmarks = tuple(
            _project_world_landmark(
                landmark=landmark,
            )
            for landmark in ((hand_world_landmarks[index] or ()) if index < len(hand_world_landmarks) else ())
        )
        hand_box = _build_square_box_from_landmarks(
            projected_landmarks,
            padding=_HAND_BOX_PADDING,
            min_context_ratio=_HAND_BOX_MIN_CONTEXT_RATIO,
        )
        roi_occupancy = None
        if hand_box is not None and roi.area > 0.0:
            roi_occupancy = max(0.0, min(1.0, hand_box.area / roi.area))

        tight_roi_frame_rgb = _crop_local_hand_from_roi_frame(
            roi_frame_rgb=roi_frame_rgb,
            local_landmarks=local_landmarks,
        )
        local_gesture_context_frame_rgb = _crop_local_hand_from_roi_frame(
            roi_frame_rgb=roi_frame_rgb,
            local_landmarks=local_landmarks,
            padding=_LOCAL_HAND_RETRY_CROP_PADDING,
            min_context_ratio=_LOCAL_HAND_RETRY_MIN_CONTEXT_RATIO,
        )
        gesture_frame_rgb = None
        gesture_context_frame_rgb = None
        if full_frame_rgb is not None:
            gesture_frame_rgb = _crop_local_hand_from_roi_frame(
                roi_frame_rgb=full_frame_rgb,
                local_landmarks=projected_landmarks,
            )
            gesture_context_frame_rgb = _crop_local_hand_from_roi_frame(
                roi_frame_rgb=full_frame_rgb,
                local_landmarks=projected_landmarks,
                padding=_LOCAL_HAND_RETRY_CROP_PADDING,
                min_context_ratio=_LOCAL_HAND_RETRY_MIN_CONTEXT_RATIO,
            )
        detections.append(
            HandLandmarkDetection(
                roi=roi,
                roi_source=roi_source,
                handedness=handedness_label,
                handedness_score=handedness_score,
                landmarks=projected_landmarks,
                roi_frame_rgb=tight_roi_frame_rgb,
                gesture_frame_rgb=gesture_frame_rgb if gesture_frame_rgb is not None else tight_roi_frame_rgb,
                gesture_context_frame_rgb=(
                    gesture_context_frame_rgb
                    if gesture_context_frame_rgb is not None
                    else local_gesture_context_frame_rgb
                ),
                world_landmarks=projected_world_landmarks,
                hand_box=hand_box,
                roi_occupancy=roi_occupancy,
                roi_priority=roi_priority,
            )
        )
    return tuple(detections)


def _crop_local_hand_from_roi_frame(
    *,
    roi_frame_rgb: Any,
    local_landmarks: Any,
    padding: float = _LOCAL_HAND_CROP_PADDING,
    min_context_ratio: float = _LOCAL_HAND_MIN_CONTEXT_RATIO,
) -> Any:
    """Return a tight hand crop inside one ROI frame when shape data is available."""

    crop_box = _build_square_box_from_landmarks(
        local_landmarks,
        padding=padding,
        min_context_ratio=min_context_ratio,
    )
    if crop_box is None:
        return roi_frame_rgb
    cropped = _crop_frame_rgb(roi_frame_rgb, crop_box)
    if cropped is None:
        return roi_frame_rgb
    return cropped


def _project_landmark_to_full_frame(*, landmark: Any, roi: AICameraBox) -> HandLandmarkPoint:
    """Map one ROI-local hand landmark back into full-frame normalized space."""

    x = roi.left + (_clamp_ratio(getattr(landmark, "x", 0.0), default=0.0) * roi.width)
    y = roi.top + (_clamp_ratio(getattr(landmark, "y", 0.0), default=0.0) * roi.height)
    score = _coerce_optional_ratio(
        getattr(landmark, "visibility", None),
        default=_coerce_optional_ratio(getattr(landmark, "presence", None), default=None),
    )
    return HandLandmarkPoint(
        x=_clamp_ratio(x, default=0.0),
        y=_clamp_ratio(y, default=0.0),
        z=_coerce_float(getattr(landmark, "z", 0.0), default=0.0),
        score=score,
    )


def _project_world_landmark(*, landmark: Any) -> HandWorldLandmarkPoint:
    """Map one MediaPipe world landmark into a typed metric point."""

    score = _coerce_optional_ratio(
        getattr(landmark, "visibility", None),
        default=_coerce_optional_ratio(getattr(landmark, "presence", None), default=None),
    )
    return HandWorldLandmarkPoint(
        x=_coerce_float(getattr(landmark, "x", 0.0), default=0.0),
        y=_coerce_float(getattr(landmark, "y", 0.0), default=0.0),
        z=_coerce_float(getattr(landmark, "z", 0.0), default=0.0),
        score=score,
    )


def _resolve_handedness(handedness: Any, *, index: int) -> tuple[str | None, float | None]:
    """Resolve the top handedness label and score for one detected hand."""

    if index >= len(handedness):
        return None, None
    categories = handedness[index] or ()
    best_label = None
    best_score = 0.0
    for category in categories:
        score = _clamp_ratio(getattr(category, "score", 0.0), default=0.0)
        if score <= best_score:
            continue
        best_score = score
        best_label = _normalize_label(getattr(category, "category_name", None))
    if best_label is None:
        return None, None
    return best_label, round(best_score, 3)


def _crop_frame_rgb(frame_rgb: Any, box: AICameraBox) -> Any | None:
    """Crop one RGB frame-like array to the requested normalized ROI box."""

    shape = getattr(frame_rgb, "shape", None)
    if not shape or len(shape) < 2:
        return None
    frame_height = int(shape[0] or 0)
    frame_width = int(shape[1] or 0)
    if frame_height <= 1 or frame_width <= 1:
        return None

    top = max(0, min(frame_height - 1, int(math.floor(box.top * frame_height))))
    left = max(0, min(frame_width - 1, int(math.floor(box.left * frame_width))))
    bottom = max(top + 1, min(frame_height, int(math.ceil(box.bottom * frame_height))))
    right = max(left + 1, min(frame_width, int(math.ceil(box.right * frame_width))))
    if bottom <= top or right <= left:
        return None
    return frame_rgb[top:bottom, left:right]


def _build_square_box_from_landmarks(
    landmarks: Any,
    *,
    padding: float,
    min_context_ratio: float,
) -> AICameraBox | None:
    """Build one clamped square box around normalized landmarks."""

    try:
        points = list(landmarks or ())
    except TypeError:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        x = _coerce_optional_ratio(getattr(point, "x", None), default=None)
        y = _coerce_optional_ratio(getattr(point, "y", None), default=None)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        return None

    raw_left = max(0.0, min(xs) - padding)
    raw_right = min(1.0, max(xs) + padding)
    raw_top = max(0.0, min(ys) - padding)
    raw_bottom = min(1.0, max(ys) + padding)
    if raw_right <= raw_left or raw_bottom <= raw_top:
        return None

    crop_width = raw_right - raw_left
    crop_height = raw_bottom - raw_top
    side = max(crop_width, crop_height, min_context_ratio)
    center_x = (raw_left + raw_right) / 2.0
    center_y = (raw_top + raw_bottom) / 2.0
    half_side = side / 2.0

    left = center_x - half_side
    right = center_x + half_side
    top = center_y - half_side
    bottom = center_y + half_side

    if left < 0.0:
        right = min(1.0, right - left)
        left = 0.0
    if right > 1.0:
        left = max(0.0, left - (right - 1.0))
        right = 1.0
    if top < 0.0:
        bottom = min(1.0, bottom - top)
        top = 0.0
    if bottom > 1.0:
        top = max(0.0, top - (bottom - 1.0))
        bottom = 1.0
    if right <= left or bottom <= top:
        return None

    return AICameraBox(top=top, left=left, bottom=bottom, right=right)


def _predict_temporal_roi_box(
    *,
    tracked_hand: _TrackedHand,
    age_ms: int,
    config: HandLandmarkWorkerConfig,
) -> AICameraBox | None:
    """Predict one ROI box from the recent hand track and simple motion."""

    center_x, center_y = _box_center(tracked_hand.hand_box)
    center_x += tracked_hand.velocity_x_per_ms * age_ms
    center_y += tracked_hand.velocity_y_per_ms * age_ms

    max_side = max(tracked_hand.hand_box.width, tracked_hand.hand_box.height)
    if max_side <= 0.0:
        return None

    age_growth = 1.0
    if config.temporal_roi_ttl_ms > 0:
        age_growth += min(0.50, (age_ms / float(config.temporal_roi_ttl_ms)) * 0.50)
    side = max_side * (1.0 + (2.0 * config.temporal_roi_padding)) * age_growth
    motion_margin = (abs(tracked_hand.velocity_x_per_ms) + abs(tracked_hand.velocity_y_per_ms)) * age_ms
    side = min(0.70, max(0.12, side + motion_margin))

    half = side / 2.0
    return AICameraBox(
        top=center_y - half,
        left=center_x - half,
        bottom=center_y + half,
        right=center_x + half,
    )


def _box_iou(first: AICameraBox, second: AICameraBox) -> float:
    """Return the intersection-over-union for two normalized boxes."""

    overlap_top = max(first.top, second.top)
    overlap_left = max(first.left, second.left)
    overlap_bottom = min(first.bottom, second.bottom)
    overlap_right = min(first.right, second.right)
    if overlap_bottom <= overlap_top or overlap_right <= overlap_left:
        return 0.0
    overlap_area = (overlap_bottom - overlap_top) * (overlap_right - overlap_left)
    union_area = first.area + second.area - overlap_area
    if union_area <= 0.0:
        return 0.0
    return overlap_area / union_area


def _should_dedupe_candidate(candidate: _HandRoiCandidate, existing: _HandRoiCandidate) -> bool:
    """Return whether two ROI candidates are redundant enough to collapse."""

    if _box_iou(candidate.box, existing.box) < 0.78:
        return False
    allowed_pair = {
        candidate.source,
        existing.source,
    } == {
        HandRoiSource.PRIMARY_PERSON_UPPER_BODY,
        HandRoiSource.PRIMARY_PERSON_FULL_BODY,
    }
    wide_allowed_pair = {
        candidate.source,
        existing.source,
    } == {
        HandRoiSource.PRIMARY_PERSON_WIDE_UPPER_BODY,
        HandRoiSource.PRIMARY_PERSON_WIDE_FULL_BODY,
    }
    return not (allowed_pair or wide_allowed_pair)


def _merge_unique_hand_detections(
    *,
    existing: list[HandLandmarkDetection],
    incoming: tuple[HandLandmarkDetection, ...],
    min_iou: float,
) -> list[HandLandmarkDetection]:
    """Merge detections that refer to the same physical hand."""

    merged = list(existing)
    for detection in incoming:
        duplicate_index = None
        for index, current in enumerate(merged):
            if _detections_represent_same_hand(current, detection, min_iou=min_iou):
                duplicate_index = index
                break
        if duplicate_index is None:
            merged.append(detection)
            continue
        better = _select_better_detection(merged[duplicate_index], detection)
        merged[duplicate_index] = better
    return merged


def _detections_represent_same_hand(
    first: HandLandmarkDetection,
    second: HandLandmarkDetection,
    *,
    min_iou: float,
) -> bool:
    """Return whether two detections likely describe the same hand."""

    first_box = first.hand_box
    second_box = second.hand_box
    if first_box is not None and second_box is not None:
        iou = _box_iou(first_box, second_box)
        if first.handedness and second.handedness and first.handedness != second.handedness and iou < 0.70:
            return False
        if iou >= min_iou:
            return True
        if iou >= 0.20 and _wrist_distance(first, second) <= 0.05:
            return True
        return False
    return _wrist_distance(first, second) <= 0.04


def _select_better_detection(
    first: HandLandmarkDetection,
    second: HandLandmarkDetection,
) -> HandLandmarkDetection:
    """Choose the better detection and backfill missing metadata from the other."""

    primary = second if _detection_sort_key(second) > _detection_sort_key(first) else first
    secondary = first if primary is second else second
    if (
        (primary.world_landmarks or not secondary.world_landmarks)
        and (primary.handedness is not None or secondary.handedness is None)
        and (primary.hand_box is not None or secondary.hand_box is None)
        and (primary.roi_occupancy is not None or secondary.roi_occupancy is None)
    ):
        return primary
    return replace(
        primary,
        world_landmarks=primary.world_landmarks or secondary.world_landmarks,
        handedness=primary.handedness if primary.handedness is not None else secondary.handedness,
        handedness_score=primary.handedness_score if primary.handedness_score is not None else secondary.handedness_score,
        hand_box=primary.hand_box or secondary.hand_box,
        roi_occupancy=primary.roi_occupancy if primary.roi_occupancy is not None else secondary.roi_occupancy,
    )


def _detection_sort_key(detection: HandLandmarkDetection) -> tuple[float, float, float, float, float, float, float]:
    """Rank detections by actual crop quality rather than raw handedness score."""

    point_scores = [point.score for point in detection.landmarks if point.score is not None]
    landmark_score = sum(point_scores) / float(len(point_scores)) if point_scores else 0.0
    occupancy = detection.roi_occupancy or 0.0
    source_preference = _source_preference(detection.roi_source)
    priority_bonus = 1.0 - min(1.0, max(0, detection.roi_priority) / 10.0)
    handedness_score = detection.handedness_score or 0.0
    world_bonus = 1.0 if detection.world_landmarks else 0.0
    smaller_roi_bonus = 1.0 - min(1.0, detection.roi.area)
    return (
        occupancy,
        landmark_score,
        source_preference,
        priority_bonus,
        handedness_score,
        world_bonus,
        smaller_roi_bonus,
    )


def _detection_rank_scalar(detection: HandLandmarkDetection) -> float:
    """Return a scalar quality score for lightweight track management."""

    key = _detection_sort_key(detection)
    return (key[0] * 100.0) + (key[1] * 10.0) + key[2] + (key[3] * 0.1) + (key[4] * 0.01) + (key[5] * 0.001)


def _find_best_track_match(
    tracked_hands: list[_TrackedHand],
    detection: HandLandmarkDetection,
) -> int | None:
    """Return the best matching track index for one new detection."""

    if detection.hand_box is None:
        return None

    best_index = None
    best_score = 0.0
    detection_center_x, detection_center_y = _box_center(detection.hand_box)
    for index, track in enumerate(tracked_hands):
        iou = _box_iou(track.hand_box, detection.hand_box)
        track_center_x, track_center_y = _box_center(track.hand_box)
        center_distance = math.hypot(detection_center_x - track_center_x, detection_center_y - track_center_y)
        score = iou - (center_distance * 0.50)
        if track.handedness and detection.handedness and track.handedness != detection.handedness:
            score -= 0.35
        if score <= best_score:
            continue
        best_score = score
        best_index = index
    if best_score < 0.10:
        return None
    return best_index


def _wrist_distance(first: HandLandmarkDetection, second: HandLandmarkDetection) -> float:
    """Return the full-frame distance between wrist landmarks when available."""

    if not first.landmarks or not second.landmarks:
        return 1.0
    return math.hypot(first.landmarks[0].x - second.landmarks[0].x, first.landmarks[0].y - second.landmarks[0].y)


def _source_preference(source: HandRoiSource) -> float:
    """Return the source preference used for detection ranking."""

    if source in {HandRoiSource.TEMPORAL_LEFT, HandRoiSource.TEMPORAL_RIGHT, HandRoiSource.TEMPORAL_UNKNOWN}:
        return _SOURCE_PREFERENCE["temporal"]
    if source in {HandRoiSource.LEFT_WRIST, HandRoiSource.RIGHT_WRIST}:
        return _SOURCE_PREFERENCE["wrist"]
    if source in {HandRoiSource.WIDE_LEFT_WRIST, HandRoiSource.WIDE_RIGHT_WRIST}:
        return _SOURCE_PREFERENCE["wide_wrist"]
    if source == HandRoiSource.PRIMARY_PERSON_UPPER_BODY:
        return _SOURCE_PREFERENCE["upper_body"]
    if source == HandRoiSource.PRIMARY_PERSON_FULL_BODY:
        return _SOURCE_PREFERENCE["full_body"]
    if source == HandRoiSource.PRIMARY_PERSON_WIDE_UPPER_BODY:
        return _SOURCE_PREFERENCE["wide_upper_body"]
    if source == HandRoiSource.PRIMARY_PERSON_WIDE_FULL_BODY:
        return _SOURCE_PREFERENCE["wide_full_body"]
    return _SOURCE_PREFERENCE["full_frame"]


def _is_wrist_source(source: HandRoiSource) -> bool:
    """Return whether one ROI source is wrist-centered."""

    return source in {
        HandRoiSource.LEFT_WRIST,
        HandRoiSource.RIGHT_WRIST,
        HandRoiSource.WIDE_LEFT_WRIST,
        HandRoiSource.WIDE_RIGHT_WRIST,
    }


def _temporal_source_for_handedness(handedness: str | None) -> HandRoiSource:
    """Resolve the temporal ROI source for one cached track."""

    if handedness == "left":
        return HandRoiSource.TEMPORAL_LEFT
    if handedness == "right":
        return HandRoiSource.TEMPORAL_RIGHT
    return HandRoiSource.TEMPORAL_UNKNOWN


def _box_center(box: AICameraBox) -> tuple[float, float]:
    """Return the normalized center of one box."""

    return ((box.left + box.right) / 2.0, (box.top + box.bottom) / 2.0)


def _normalize_label(value: object) -> str:
    """Normalize one classifier label into a stable lowercase token."""

    return "_".join(str(value or "").strip().lower().split())


def _distance_2d(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    """Return the normalized 2D distance between two keypoints."""

    return math.hypot(first[0] - second[0], first[1] - second[1])


def _normalize_vector_2d(x: float, y: float) -> tuple[float, float]:
    """Normalize one 2D vector."""

    norm = math.hypot(x, y)
    if norm <= 1e-9:
        return 0.0, 0.0
    return x / norm, y / norm


def _require_runtime_item(runtime: dict[str, Any], key: str) -> Any:
    """Resolve one required runtime dependency with a stable error message."""

    if key not in runtime:
        raise KeyError(f"mediapipe_runtime_dependency_missing:{key}")
    return runtime[key]


def _clamp_ratio(value: object, *, default: float) -> float:
    """Clamp one numeric value into the unit interval."""

    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _coerce_float(value: object, *, default: float) -> float:
    """Coerce one value into a finite float."""

    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _coerce_optional_ratio(value: object, *, default: float | None) -> float | None:
    """Coerce one optional ratio into the unit interval."""

    if value is None:
        return default
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _coerce_int(
    value: object,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Coerce one value into a bounded integer."""

    try:
        number = int(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if minimum is not None and number < minimum:
        return minimum
    if maximum is not None and number > maximum:
        return maximum
    return number


def _coerce_bool(value: object, *, default: bool) -> bool:
    """Coerce one value into a boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


__all__ = [
    "DEFAULT_MEDIAPIPE_HAND_LANDMARKER_MODEL_URL",
    "HandLandmarkDetection",
    "HandLandmarkPoint",
    "HandLandmarkResult",
    "HandLandmarkWorkerConfig",
    "HandRoiSource",
    "HandWorldLandmarkPoint",
    "MediaPipeHandLandmarkWorker",
    "_build_hand_roi_candidates",
    "_project_landmark_to_full_frame",
]