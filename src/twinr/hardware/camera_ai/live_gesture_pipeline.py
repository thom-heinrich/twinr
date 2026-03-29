# CHANGELOG: 2026-03-28
# BUG-1: Fixed async result correlation so observe() only treats exact-frame callbacks as "current"; future-frame callbacks are no longer misattributed to earlier frames.
# BUG-2: Fixed stale hand-box rescue leakage after fresh live no-hand results; recent cached boxes can no longer bypass the current-frame no-hand veto.
# BUG-3: Enforced monotonically increasing LIVE_STREAM timestamps before recognize_async() to avoid MediaPipe timestamp regressions and silent frame drops.
# BUG-4: Fixed the logically dead wave path; wave tracking now consumes current-frame open-palm center evidence directly instead of requiring an allowlisted fine-hand label.
# BUG-5: Live-stream callbacks now carry stable runtime cache keys, so per-frame generation closures no longer force recognizer recreation.
# SEC-1: Added bounded person-ROI / hand-ROI candidate budgets and full-frame rescue rate limiting to reduce practical CPU-exhaustion and latency-collapse attacks on Raspberry Pi 4.
# SEC-2: Added generation-guarded callbacks and fail-closed exception handling so late callbacks after close() and recognizer/runtime errors do not corrupt live state or crash the lane.
# IMP-1: Added bounded per-timestamp snapshot history so fallback reads exact-or-older results but never "future" callbacks from later frames.
# IMP-2: Added lightweight temporal stabilization for rescue paths plus wave cooldown/reset logic to reduce flicker and false positives in continuous user-facing HCI.
# IMP-3: Added IoU-based hand-box deduplication and config-driven tuning knobs via getattr(config, ...) without breaking older configs.

"""Run a dedicated low-latency MediaPipe live-stream gesture lane.

Twinr's user-facing HDMI gesture acknowledgements should not depend on the
general social-vision pipeline that also performs person detection, pose
classification, and temporal room reasoning. This module keeps a much thinner
hot path for the gestures that must feel instant to a person standing in front
of the Pi: capture one RGB frame, feed it into MediaPipe's live-stream gesture
recognizers, keep the newest callback result, and expose a compact
`LiveGestureFrameObservation` for downstream acknowledgement logic.

The Pi live HCI path is intentionally narrowed to the three gestures that
matter most for current device control: `thumbs_up`, `thumbs_down`, and
`peace_sign`. Other labels may still exist in broader offline/social surfaces,
but they are not accepted by this low-latency user-facing lane.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Condition, RLock
from time import monotonic
from typing import Any, cast
import math

from twinr.agent.workflows.forensics import workflow_decision, workflow_event, workflow_span
from twinr.hardware.hand_landmarks import (
    HandLandmarkWorkerConfig,
    HandRoiSource,
    MediaPipeHandLandmarkWorker,
)

from .authoritative_gestures import AuthoritativeGestureLane
from .config import MediaPipeVisionConfig
from .fine_hand_gestures import (
    BUILTIN_FINE_GESTURE_MAP,
    CUSTOM_FINE_GESTURE_MAP,
    combine_task_specific_custom_gesture_choice,
    normalize_category_name,
    prefer_gesture_choice,
    resolve_fine_hand_gesture,
)
from .geometry import iou
from .mediapipe_runtime import MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR, MediaPipeTaskRuntime
from .models import AICameraBox, AICameraFineHandGesture, AICameraGestureEvent


_DEFAULT_MAX_RESULT_AGE_S = 0.75
_DEFAULT_CURRENT_RESULT_WAIT_S = 0.20
_DEFAULT_RESULT_HISTORY_SIZE = 12
_DEFAULT_MAX_PENDING_LIVE_RESULT_AGE_S = 1.0
_LIVE_CUSTOM_MIN_SCORE_FLOOR = 0.60

_WAVE_WINDOW_S = 0.55
_WAVE_MIN_SAMPLES = 3
_WAVE_MIN_SPAN_X = 0.10
_WAVE_MIN_TRAVEL_X = 0.18
_WAVE_DIRECTION_DELTA = 0.015
_WAVE_MAX_GAP_S = 0.16
_WAVE_COOLDOWN_S = 0.70

_RECENT_PRIMARY_PERSON_BOX_TTL_S = 0.45
_RECENT_VISIBLE_PERSON_BOXES_TTL_S = 0.45
_RECENT_HAND_BOXES_TTL_S = 0.45

_LIVE_GESTURE_NUM_HANDS = 1
_PRIMARY_PERSON_HINT_MIN_IOU = 0.60

_LIVE_HAND_CROP_PADDING = 0.14
_LIVE_HAND_MIN_CONTEXT_RATIO = 0.18
_HAND_BOX_DUPLICATE_IOU = 0.62

_PERSON_ROI_SOURCE_PRIORITY_MARGIN = 0.12
_PERSON_ROI_POSE_HINT_SHORT_CIRCUIT_REASON = "primary_pose_hint_confident_gesture"

_DEFAULT_MAX_PERSON_ROI_CANDIDATES = 3
_DEFAULT_MAX_HAND_ROI_CANDIDATES = 2
_DEFAULT_FULL_FRAME_RESCUE_MIN_INTERVAL_S = 0.18

_TEMPORAL_WINDOW_S = 0.35
_TEMPORAL_WINDOW_SAMPLES = 3
_TEMPORAL_REQUIRED_VOTES = 2
_TEMPORAL_STRONG_CONFIDENCE = 0.84
_TEMPORAL_HOLD_S = 0.22

_PERSON_ROI_SOURCE_PRIORITY = {
    HandRoiSource.FULL_FRAME.value: 5,
    HandRoiSource.LEFT_WRIST.value: 4,
    HandRoiSource.RIGHT_WRIST.value: 4,
    HandRoiSource.WIDE_LEFT_WRIST.value: 3,
    HandRoiSource.WIDE_RIGHT_WRIST.value: 3,
    HandRoiSource.PRIMARY_PERSON_UPPER_BODY.value: 2,
    HandRoiSource.PRIMARY_PERSON_FULL_BODY.value: 1,
    HandRoiSource.PRIMARY_PERSON_WIDE_UPPER_BODY.value: 1,
    HandRoiSource.PRIMARY_PERSON_WIDE_FULL_BODY.value: 0,
}
_PERSON_ROI_SOURCE_MIN_CONFIDENCE = {
    HandRoiSource.LEFT_WRIST.value: 0.56,
    HandRoiSource.RIGHT_WRIST.value: 0.56,
    HandRoiSource.WIDE_LEFT_WRIST.value: 0.62,
    HandRoiSource.WIDE_RIGHT_WRIST.value: 0.62,
    HandRoiSource.PRIMARY_PERSON_UPPER_BODY.value: 0.76,
    HandRoiSource.PRIMARY_PERSON_FULL_BODY.value: 0.82,
    HandRoiSource.PRIMARY_PERSON_WIDE_UPPER_BODY.value: 0.82,
    HandRoiSource.PRIMARY_PERSON_WIDE_FULL_BODY.value: 0.88,
}
_LIVE_FINE_GESTURE_ALLOWLIST = frozenset(
    {
        AICameraFineHandGesture.THUMBS_UP,
        AICameraFineHandGesture.THUMBS_DOWN,
        AICameraFineHandGesture.PEACE_SIGN,
    }
)
_LIVE_NO_HAND_PERSON_ROI_ALLOWLIST = frozenset({AICameraFineHandGesture.PEACE_SIGN})
_LIVE_BUILTIN_FINE_GESTURE_MAP = {
    label: gesture
    for label, gesture in BUILTIN_FINE_GESTURE_MAP.items()
    if gesture in _LIVE_FINE_GESTURE_ALLOWLIST
}
_LIVE_CUSTOM_FINE_GESTURE_MAP = {
    label: gesture
    for label, gesture in CUSTOM_FINE_GESTURE_MAP.items()
    if gesture in _LIVE_FINE_GESTURE_ALLOWLIST
}
_HandBox = tuple[float, float, float, float]
_GestureChoice = tuple[AICameraFineHandGesture, float | None]


@dataclass(frozen=True, slots=True)
class LiveGestureFrameObservation:
    """Describe one low-latency live-stream gesture observation."""

    observed_at: float
    fine_hand_gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    fine_hand_gesture_confidence: float | None = None
    gesture_event: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_confidence: float | None = None
    hand_count: int = 0
    result_age_s: float | None = None
    gesture_temporal_authoritative: bool = False
    gesture_activation_key: str | None = None
    gesture_activation_token: int | None = None
    gesture_activation_started_at: float | None = None
    gesture_activation_changed_at: float | None = None
    gesture_activation_source: str | None = None
    gesture_activation_rising: bool = False


@dataclass(frozen=True, slots=True)
class LiveGestureObservePolicy:
    """Describe which recovery stages one observe call may use."""

    name: str = "full"
    enable_custom_live: bool = True
    enable_custom_roi: bool = True
    allow_custom_recovery_when_live_custom_pending: bool = False
    allow_person_roi_recovery: bool = True
    allow_live_hand_roi_recovery: bool = True
    allow_full_frame_hand_recovery: bool = True
    allow_full_frame_hand_recovery_after_person_roi_detection: bool = False
    allow_recent_person_boxes: bool = True
    allow_recent_hand_boxes: bool = True
    prefer_live_hand_roi_before_person_roi: bool = False
    require_weak_live_stream_consensus: bool = False

    @classmethod
    def full(cls) -> "LiveGestureObservePolicy":
        return cls()

    @classmethod
    def user_facing_fast(cls) -> "LiveGestureObservePolicy":
        return cls(
            name="user_facing_fast",
            enable_custom_live=True,
            enable_custom_roi=False,
            allow_custom_recovery_when_live_custom_pending=True,
            allow_person_roi_recovery=True,
            allow_live_hand_roi_recovery=True,
            allow_full_frame_hand_recovery=False,
            allow_full_frame_hand_recovery_after_person_roi_detection=True,
            allow_recent_person_boxes=False,
            allow_recent_hand_boxes=False,
            prefer_live_hand_roi_before_person_roi=True,
            require_weak_live_stream_consensus=True,
        )


@dataclass(frozen=True, slots=True)
class _RecognizerSnapshot:
    """Store one live-stream recognizer callback payload."""

    timestamp_ms: int
    gesture: AICameraFineHandGesture
    confidence: float | None
    hand_count: int
    open_palm_center_x: float | None
    hand_boxes: tuple[_HandBox, ...] = ()


@dataclass(frozen=True, slots=True)
class _WaveSample:
    """Store one open-palm motion sample for wave tracking."""

    observed_at: float
    center_x: float


@dataclass(frozen=True, slots=True)
class _TemporalGestureSample:
    """Store one final-stage gesture sample for temporal stabilization."""

    observed_at: float
    gesture: AICameraFineHandGesture
    confidence: float | None
    resolved_source: str
    hand_count: int


@dataclass(frozen=True, slots=True)
class _PersonRoiGestureChoice:
    """Store one person-ROI gesture candidate with its source provenance."""

    gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    confidence: float | None = None
    roi_source: str = ""
    detection_index: int | None = None

    def as_choice(self) -> _GestureChoice:
        return self.gesture, self.confidence


class _WaveGestureTracker:
    """Detect a short open-palm lateral wave over recent callback samples."""

    def __init__(self) -> None:
        self._samples: deque[_WaveSample] = deque()
        self._last_open_palm_at: float | None = None
        self._last_emitted_at: float | None = None

    def observe(
        self,
        *,
        observed_at: float,
        open_palm_center_x: float | None,
    ) -> tuple[AICameraGestureEvent, float | None]:
        self._prune(observed_at)
        if open_palm_center_x is None:
            if (
                self._last_open_palm_at is not None
                and (observed_at - self._last_open_palm_at) > _WAVE_MAX_GAP_S
            ):
                self._samples.clear()
            return AICameraGestureEvent.NONE, None

        if (
            self._last_open_palm_at is not None
            and (observed_at - self._last_open_palm_at) > _WAVE_MAX_GAP_S
        ):
            self._samples.clear()
        self._last_open_palm_at = observed_at
        self._samples.append(_WaveSample(observed_at=observed_at, center_x=open_palm_center_x))
        self._prune(observed_at)
        if len(self._samples) < _WAVE_MIN_SAMPLES:
            return AICameraGestureEvent.NONE, None

        xs = [sample.center_x for sample in self._samples]
        span_x = max(xs) - min(xs)
        total_travel_x = 0.0
        direction_changes = 0
        last_sign = 0
        for previous, current in zip(xs, xs[1:]):
            delta = current - previous
            total_travel_x += abs(delta)
            if abs(delta) < _WAVE_DIRECTION_DELTA:
                continue
            sign = 1 if delta > 0.0 else -1
            if last_sign != 0 and sign != last_sign:
                direction_changes += 1
            last_sign = sign

        if span_x < _WAVE_MIN_SPAN_X or total_travel_x < _WAVE_MIN_TRAVEL_X or direction_changes < 1:
            return AICameraGestureEvent.NONE, None
        if self._last_emitted_at is not None and (observed_at - self._last_emitted_at) < _WAVE_COOLDOWN_S:
            return AICameraGestureEvent.NONE, None

        self._last_emitted_at = observed_at
        self._samples.clear()
        confidence = min(
            0.99,
            0.55 + min(0.25, span_x * 1.2) + min(0.19, total_travel_x * 0.6),
        )
        return AICameraGestureEvent.WAVE, round(confidence, 3)

    def _prune(self, observed_at: float) -> None:
        cutoff = observed_at - _WAVE_WINDOW_S
        while self._samples and self._samples[0].observed_at < cutoff:
            self._samples.popleft()


class _GestureStabilityTracker:
    """Apply a tiny streaming stabilizer without adding a heavy temporal model."""

    def __init__(self, *, config: MediaPipeVisionConfig) -> None:
        self._config = config
        self._samples: deque[_TemporalGestureSample] = deque()
        self._last_output_gesture = AICameraFineHandGesture.NONE
        self._last_output_confidence: float | None = None
        self._last_output_at: float | None = None

    def observe(
        self,
        *,
        observed_at: float,
        gesture: AICameraFineHandGesture,
        confidence: float | None,
        resolved_source: str,
        hand_count: int,
        require_weak_live_stream_consensus: bool = False,
    ) -> tuple[AICameraFineHandGesture, float | None, dict[str, object]]:
        temporal_stability_enabled = _live_temporal_stability_enabled(self._config)
        temporal_enabled = temporal_stability_enabled or require_weak_live_stream_consensus
        if not temporal_enabled:
            return gesture, confidence, {
                "temporal_enabled": False,
                "temporal_reason": "disabled",
                "temporal_output_gesture": gesture.value,
                "temporal_output_confidence": _round_optional_confidence(confidence),
                "temporal_output_changed": False,
                "temporal_resolved_source": resolved_source,
            }

        self._samples.append(
            _TemporalGestureSample(
                observed_at=observed_at,
                gesture=gesture,
                confidence=confidence,
                resolved_source=resolved_source,
                hand_count=max(0, int(hand_count)),
            )
        )
        self._prune(observed_at)
        strong_confidence = _live_temporal_strong_confidence(self._config)
        hold_s = _live_temporal_hold_s(self._config)
        score = _coerce_unit_interval(confidence) or 0.0

        if gesture != AICameraFineHandGesture.NONE:
            votes, mean_confidence = self._votes_for_gesture(gesture)
            if score >= strong_confidence:
                self._remember_output(gesture, confidence, observed_at)
                return gesture, confidence, {
                    "temporal_enabled": temporal_enabled,
                    "temporal_reason": "strong_current",
                    "temporal_votes": votes,
                    "temporal_output_gesture": gesture.value,
                    "temporal_output_confidence": _round_optional_confidence(confidence),
                    "temporal_output_changed": False,
                    "temporal_resolved_source": resolved_source,
                }
            if votes >= _live_temporal_required_votes(self._config):
                stabilized_confidence = max(score, mean_confidence or 0.0)
                self._remember_output(gesture, stabilized_confidence, observed_at)
                return gesture, stabilized_confidence, {
                    "temporal_enabled": temporal_enabled,
                    "temporal_reason": "consensus",
                    "temporal_votes": votes,
                    "temporal_output_gesture": gesture.value,
                    "temporal_output_confidence": _round_optional_confidence(stabilized_confidence),
                    "temporal_output_changed": False,
                    "temporal_resolved_source": "temporal_consensus",
                }
            if resolved_source != "live_stream":
                if not temporal_stability_enabled:
                    return gesture, confidence, {
                        "temporal_enabled": temporal_enabled,
                        "temporal_reason": "rescue_passthrough",
                        "temporal_votes": votes,
                        "temporal_output_gesture": gesture.value,
                        "temporal_output_confidence": _round_optional_confidence(confidence),
                        "temporal_output_changed": False,
                        "temporal_resolved_source": resolved_source,
                    }
                return AICameraFineHandGesture.NONE, None, {
                    "temporal_enabled": temporal_enabled,
                    "temporal_reason": "rescue_waiting_for_consensus",
                    "temporal_votes": votes,
                    "temporal_output_gesture": AICameraFineHandGesture.NONE.value,
                    "temporal_output_confidence": None,
                    "temporal_output_changed": True,
                    "temporal_resolved_source": "temporal_guard",
                }
            if require_weak_live_stream_consensus:
                return AICameraFineHandGesture.NONE, None, {
                    "temporal_enabled": temporal_enabled,
                    "temporal_reason": "weak_live_waiting_for_consensus",
                    "temporal_votes": votes,
                    "temporal_output_gesture": AICameraFineHandGesture.NONE.value,
                    "temporal_output_confidence": None,
                    "temporal_output_changed": True,
                    "temporal_resolved_source": "temporal_guard",
                }
            self._remember_output(gesture, confidence, observed_at)
            return gesture, confidence, {
                "temporal_enabled": temporal_enabled,
                "temporal_reason": "live_passthrough",
                "temporal_votes": votes,
                "temporal_output_gesture": gesture.value,
                "temporal_output_confidence": _round_optional_confidence(confidence),
                "temporal_output_changed": False,
                "temporal_resolved_source": resolved_source,
            }

        if (
            self._last_output_gesture != AICameraFineHandGesture.NONE
            and self._last_output_at is not None
            and (observed_at - self._last_output_at) <= hold_s
            and (hand_count > 0 or (observed_at - self._last_output_at) <= (hold_s * 0.5))
        ):
            return self._last_output_gesture, self._last_output_confidence, {
                "temporal_enabled": temporal_enabled,
                "temporal_reason": "hold_recent",
                "temporal_votes": 0,
                "temporal_output_gesture": self._last_output_gesture.value,
                "temporal_output_confidence": _round_optional_confidence(self._last_output_confidence),
                "temporal_output_changed": True,
                "temporal_resolved_source": "temporal_hold",
            }

        return gesture, confidence, {
            "temporal_enabled": temporal_enabled,
            "temporal_reason": "none",
            "temporal_votes": 0,
            "temporal_output_gesture": gesture.value,
            "temporal_output_confidence": _round_optional_confidence(confidence),
            "temporal_output_changed": False,
            "temporal_resolved_source": resolved_source,
        }

    def _votes_for_gesture(self, gesture: AICameraFineHandGesture) -> tuple[int, float | None]:
        votes = 0
        confidences: list[float] = []
        for sample in list(self._samples)[-_live_temporal_window_samples(self._config):]:
            if sample.gesture != gesture:
                continue
            votes += 1
            sample_score = _coerce_unit_interval(sample.confidence)
            if sample_score is not None:
                confidences.append(sample_score)
        if not confidences:
            return votes, None
        return votes, (sum(confidences) / len(confidences))

    def _remember_output(
        self,
        gesture: AICameraFineHandGesture,
        confidence: float | None,
        observed_at: float,
    ) -> None:
        self._last_output_gesture = gesture
        self._last_output_confidence = confidence
        self._last_output_at = observed_at

    def _prune(self, observed_at: float) -> None:
        cutoff = observed_at - _live_temporal_window_s(self._config)
        while self._samples and self._samples[0].observed_at < cutoff:
            self._samples.popleft()
        while len(self._samples) > _live_temporal_window_samples(self._config):
            self._samples.popleft()


class LiveGesturePipeline:
    """Submit RGB frames to MediaPipe LIVE_STREAM gesture recognizers."""

    def __init__(self, *, config: MediaPipeVisionConfig) -> None:
        self.config = config
        self._runtime = MediaPipeTaskRuntime(config=config)
        self._lock = RLock()
        self._result_condition = Condition(self._lock)

        self._latest_builtin: _RecognizerSnapshot | None = None
        self._latest_custom: _RecognizerSnapshot | None = None
        self._builtin_history: OrderedDict[int, _RecognizerSnapshot] = OrderedDict()
        self._custom_history: OrderedDict[int, _RecognizerSnapshot] = OrderedDict()

        self._wave_tracker = _WaveGestureTracker()
        self._stability_tracker = _GestureStabilityTracker(config=config)
        self._authoritative_gesture_lane = AuthoritativeGestureLane()
        self._hand_landmark_worker: MediaPipeHandLandmarkWorker | None = None

        self._recent_primary_person_box: AICameraBox | None = None
        self._recent_primary_person_seen_at: float | None = None
        self._recent_visible_person_boxes: tuple[AICameraBox, ...] = ()
        self._recent_visible_person_boxes_seen_at: float | None = None
        self._recent_hand_boxes: tuple[_HandBox, ...] = ()
        self._recent_hand_boxes_seen_at: float | None = None

        self._last_debug_snapshot: dict[str, object] = {}
        self._last_submitted_live_timestamp_ms = 0
        self._last_full_frame_rescue_at: float | None = None
        self._pending_live_builtin_timestamp_ms: int | None = None
        self._pending_live_custom_timestamp_ms: int | None = None
        self._pending_live_builtin_submitted_mono_s: float | None = None
        self._pending_live_custom_submitted_mono_s: float | None = None
        self._callback_generation = 0
        self._builtin_callback_cache_key = ("LiveGesturePipeline", id(self), "builtin")
        self._custom_callback_cache_key = ("LiveGesturePipeline", id(self), "custom")

    def close(self) -> None:
        """Close live-stream recognizers and reset cached callback state."""

        with self._lock:
            self._callback_generation += 1
            self._latest_builtin = None
            self._latest_custom = None
            self._builtin_history.clear()
            self._custom_history.clear()
            self._wave_tracker = _WaveGestureTracker()
            self._stability_tracker = _GestureStabilityTracker(config=self.config)
            self._authoritative_gesture_lane.reset()
            self._last_submitted_live_timestamp_ms = 0
            self._last_full_frame_rescue_at = None
            self._pending_live_builtin_timestamp_ms = None
            self._pending_live_custom_timestamp_ms = None
            self._pending_live_builtin_submitted_mono_s = None
            self._pending_live_custom_submitted_mono_s = None

            worker = self._hand_landmark_worker
            self._hand_landmark_worker = None

            self._recent_primary_person_box = None
            self._recent_primary_person_seen_at = None
            self._recent_visible_person_boxes = ()
            self._recent_visible_person_boxes_seen_at = None
            self._recent_hand_boxes = ()
            self._recent_hand_boxes_seen_at = None
            self._last_debug_snapshot = {}

        if worker is not None:
            worker.close()
        self._runtime.close()

    def debug_snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._last_debug_snapshot)

    def observe(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
        observe_policy: LiveGestureObservePolicy | None = None,
        primary_person_box: AICameraBox | None = None,
        visible_person_boxes: tuple[AICameraBox, ...] = (),
        person_count: int = 0,
        sparse_keypoints: dict[int, tuple[float, float, float]] | None = None,
    ) -> LiveGestureFrameObservation:
        """Submit one RGB frame and return the freshest bounded live result."""

        active_policy = observe_policy or LiveGestureObservePolicy.full()
        with workflow_span(
            name="live_gesture_pipeline_observe",
            kind="io",
            details={
                "person_count": max(0, int(person_count)),
                "visible_person_box_count": len(tuple(visible_person_boxes or ())),
                "pose_hint_keypoint_count": len(dict(sparse_keypoints or {})),
                "observe_policy": active_policy.name,
            },
        ):
            try:
                runtime = self._runtime.load_runtime()
                image = self._runtime.build_image(runtime, frame_rgb=frame_rgb)
                timestamp_ms = self._next_live_timestamp_ms(observed_at)
                frame_time_s = timestamp_ms / 1000.0
                observe_started_mono_s = monotonic()
                live_custom_enabled = (
                    active_policy.enable_custom_live and _live_custom_gesture_enabled(self.config)
                )
                live_submission_debug = self._reset_stale_live_submissions(
                    observed_mono_s=observe_started_mono_s,
                )
                with self._lock:
                    generation = self._callback_generation

                builtin_submitted = self._submit_live_builtin_frame(
                    runtime=runtime,
                    image=image,
                    timestamp_ms=timestamp_ms,
                    generation=generation,
                    observed_mono_s=observe_started_mono_s,
                )

                custom_submitted = False
                if live_custom_enabled:
                    custom_submitted = self._submit_live_custom_frame(
                        runtime=runtime,
                        image=image,
                        timestamp_ms=timestamp_ms,
                        generation=generation,
                        observed_mono_s=observe_started_mono_s,
                    )

                wait_s, builtin_ready, custom_ready = self._await_current_live_results(
                    timestamp_ms=timestamp_ms,
                    expect_builtin=builtin_submitted,
                    expect_custom=live_custom_enabled and custom_submitted,
                )

                return self._build_observation(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    observed_at=observed_at,
                    frame_time_s=frame_time_s,
                    timestamp_ms=timestamp_ms,
                    live_result_wait_s=wait_s,
                    current_live_builtin_ready=builtin_ready,
                    current_live_custom_ready=(custom_ready if live_custom_enabled else None),
                    live_builtin_submitted=builtin_submitted,
                    live_custom_submitted=(custom_submitted if live_custom_enabled else None),
                    live_submission_debug=live_submission_debug,
                    observe_policy=active_policy,
                    primary_person_box=primary_person_box,
                    visible_person_boxes=visible_person_boxes,
                    person_count=person_count,
                    sparse_keypoints=dict(sparse_keypoints or {}),
                )
            except Exception as exc:
                return self._fail_closed_observation(
                    observed_at=observed_at,
                    observe_policy=active_policy,
                    person_count=person_count,
                    visible_person_boxes=visible_person_boxes,
                    sparse_keypoints=dict(sparse_keypoints or {}),
                    stage="observe",
                    exc=exc,
                )

    def _build_observation(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        observed_at: float,
        frame_time_s: float,
        timestamp_ms: int,
        live_result_wait_s: float,
        current_live_builtin_ready: bool,
        current_live_custom_ready: bool | None,
        live_builtin_submitted: bool,
        live_custom_submitted: bool | None,
        live_submission_debug: Mapping[str, object],
        observe_policy: LiveGestureObservePolicy,
        primary_person_box: AICameraBox | None,
        visible_person_boxes: tuple[AICameraBox, ...],
        person_count: int,
        sparse_keypoints: dict[int, tuple[float, float, float]],
    ) -> LiveGestureFrameObservation:
        with self._lock:
            current_builtin = self._builtin_history.get(timestamp_ms)
            current_custom = self._custom_history.get(timestamp_ms)
            builtin = self._select_snapshot_from_history(
                history=self._builtin_history,
                max_timestamp_ms=timestamp_ms,
                frame_time_s=frame_time_s,
            )
            custom = self._select_snapshot_from_history(
                history=self._custom_history,
                max_timestamp_ms=timestamp_ms,
                frame_time_s=frame_time_s,
            )

            builtin_choice = (
                (AICameraFineHandGesture.NONE, None)
                if builtin is None
                else (builtin.gesture, builtin.confidence)
            )
            custom_choice = (
                (AICameraFineHandGesture.NONE, None)
                if custom is None
                else (custom.gesture, custom.confidence)
            )
            fine_hand_gesture, fine_hand_confidence = combine_task_specific_custom_gesture_choice(
                builtin_choice,
                custom_choice,
                preferred_custom_gestures=_LIVE_FINE_GESTURE_ALLOWLIST,
            )
            hand_count = max(
                0,
                0 if builtin is None else builtin.hand_count,
                0 if custom is None else custom.hand_count,
            )

            open_palm_center_x = None if current_builtin is None else current_builtin.open_palm_center_x
            gesture_event, gesture_confidence = self._wave_tracker.observe(
                observed_at=observed_at,
                open_palm_center_x=open_palm_center_x,
            )

            freshest_timestamp_ms = max(
                0 if builtin is None else builtin.timestamp_ms,
                0 if custom is None else custom.timestamp_ms,
            )
            result_age_s = (
                None
                if freshest_timestamp_ms <= 0
                else max(0.0, frame_time_s - (freshest_timestamp_ms / 1000.0))
            )

            current_live_hand_boxes = _merge_hand_boxes(
                () if current_builtin is None else current_builtin.hand_boxes,
                () if current_custom is None else current_custom.hand_boxes,
            )
            current_live_hand_count_exact = max(
                0,
                0 if current_builtin is None else current_builtin.hand_count,
                0 if current_custom is None else current_custom.hand_count,
            )

            live_visible_person_boxes = tuple(box for box in visible_person_boxes if box is not None)
            if primary_person_box is not None and person_count > 0:
                self._recent_primary_person_box = primary_person_box
                self._recent_primary_person_seen_at = observed_at
            if live_visible_person_boxes and person_count > 0:
                self._recent_visible_person_boxes = live_visible_person_boxes
                self._recent_visible_person_boxes_seen_at = observed_at

            effective_primary_person_box = primary_person_box
            primary_person_box_source = "live" if primary_person_box is not None and person_count > 0 else "none"
            if effective_primary_person_box is None and observe_policy.allow_recent_person_boxes:
                effective_primary_person_box = self._fresh_recent_primary_person_box(observed_at)
                if effective_primary_person_box is not None:
                    primary_person_box_source = "recent"

            effective_visible_person_boxes = live_visible_person_boxes
            visible_person_box_source = "live" if live_visible_person_boxes and person_count > 0 else "none"
            if not effective_visible_person_boxes and observe_policy.allow_recent_person_boxes:
                effective_visible_person_boxes = self._fresh_recent_visible_person_boxes(observed_at)
                if effective_visible_person_boxes:
                    visible_person_box_source = "recent"
            if not effective_visible_person_boxes and effective_primary_person_box is not None:
                effective_visible_person_boxes = (effective_primary_person_box,)
                visible_person_box_source = primary_person_box_source

            if current_live_hand_boxes:
                self._recent_hand_boxes = tuple(current_live_hand_boxes)
                self._recent_hand_boxes_seen_at = observed_at
                effective_hand_boxes = tuple(current_live_hand_boxes)
                hand_box_source = "live"
            else:
                effective_hand_boxes = (
                    self._fresh_recent_hand_boxes(observed_at)
                    if observe_policy.allow_recent_hand_boxes
                    else ()
                )
                hand_box_source = "recent" if effective_hand_boxes else "none"

        fresh_live_results_confirm_no_hand = _fresh_live_results_confirm_no_hand(
            builtin_ready=current_live_builtin_ready,
            custom_ready=current_live_custom_ready,
            hand_count=current_live_hand_count_exact,
            live_hand_box_count=len(current_live_hand_boxes),
        )
        live_custom_enabled = (
            observe_policy.enable_custom_live and _live_custom_gesture_enabled(self.config)
        )
        sync_custom_recovery_enabled, sync_custom_recovery_reason = _resolve_sync_custom_recovery_mode(
            observe_policy=observe_policy,
            live_custom_enabled=live_custom_enabled,
            current_live_custom_ready=current_live_custom_ready,
        )
        deferred_live_stream_choice: _GestureChoice | None = None
        if _should_defer_live_stream_candidate_for_sync_custom_recovery(
            config=self.config,
            observe_policy=observe_policy,
            sync_custom_recovery_enabled=sync_custom_recovery_enabled,
            gesture=fine_hand_gesture,
            confidence=fine_hand_confidence,
        ):
            deferred_live_stream_choice = (fine_hand_gesture, fine_hand_confidence)
            fine_hand_gesture = AICameraFineHandGesture.NONE
            fine_hand_confidence = None

        debug_snapshot: dict[str, object] = {
            "resolved_source": "none" if fine_hand_gesture == AICameraFineHandGesture.NONE else "live_stream",
            "live_builtin_gesture": builtin_choice[0].value,
            "live_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
            "live_custom_gesture": custom_choice[0].value,
            "live_custom_confidence": _round_optional_confidence(custom_choice[1]),
            "live_custom_enabled": live_custom_enabled,
            "live_hand_count": hand_count,
            "live_hand_count_exact": current_live_hand_count_exact,
            "live_hand_box_count": len(current_live_hand_boxes),
            "effective_live_hand_box_count": len(effective_hand_boxes),
            "live_hand_box_source": hand_box_source,
            "live_result_age_s": _round_optional_confidence(result_age_s),
            "current_live_result_wait_s": _round_optional_confidence(live_result_wait_s),
            "current_live_builtin_ready": current_live_builtin_ready,
            "current_live_custom_ready": current_live_custom_ready,
            "live_builtin_submitted": live_builtin_submitted,
            "live_custom_submitted": live_custom_submitted,
            "fresh_live_results_confirm_no_hand": fresh_live_results_confirm_no_hand,
            "input_person_count": max(0, int(person_count)),
            "primary_person_box_available": primary_person_box is not None,
            "effective_primary_person_box_available": effective_primary_person_box is not None,
            "primary_person_box_source": primary_person_box_source,
            "pose_hint_keypoint_count": len(sparse_keypoints),
            "visible_person_box_count": len(live_visible_person_boxes),
            "effective_visible_person_box_count": len(effective_visible_person_boxes),
            "visible_person_box_source": visible_person_box_source,
            "person_roi_candidate_count_total": len(effective_visible_person_boxes),
            "person_roi_candidate_count": 0,
            "person_roi_detection_count": 0,
            "person_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_combined_confidence": None,
            "person_roi_block_reason": None,
            "person_roi_short_circuit_used": False,
            "live_roi_candidate_count_total": len(effective_hand_boxes),
            "live_roi_candidate_count": 0,
            "live_roi_hand_box_count": 0,
            "live_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_combined_confidence": None,
            "live_roi_skip_reason": None,
            "full_frame_hand_attempt_reason": None,
            "observe_policy": observe_policy.name,
            "observe_policy_enable_custom_live": observe_policy.enable_custom_live,
            "observe_policy_enable_custom_roi": observe_policy.enable_custom_roi,
            "observe_policy_allow_custom_recovery_when_live_custom_pending": (
                observe_policy.allow_custom_recovery_when_live_custom_pending
            ),
            "observe_policy_allow_person_roi_recovery": observe_policy.allow_person_roi_recovery,
            "observe_policy_allow_live_hand_roi_recovery": observe_policy.allow_live_hand_roi_recovery,
            "observe_policy_allow_full_frame_hand_recovery": observe_policy.allow_full_frame_hand_recovery,
            "observe_policy_allow_full_frame_after_person_roi_detection": (
                observe_policy.allow_full_frame_hand_recovery_after_person_roi_detection
            ),
            "observe_policy_allow_recent_person_boxes": observe_policy.allow_recent_person_boxes,
            "observe_policy_allow_recent_hand_boxes": observe_policy.allow_recent_hand_boxes,
            "observe_policy_require_weak_live_stream_consensus": (
                observe_policy.require_weak_live_stream_consensus
            ),
            "sync_custom_recovery_enabled": sync_custom_recovery_enabled,
            "sync_custom_recovery_reason": sync_custom_recovery_reason,
            "live_stream_recovery_gate": (
                "deferred_for_sync_custom_recovery"
                if deferred_live_stream_choice is not None
                else "not_needed"
            ),
        }
        debug_snapshot.update(dict(live_submission_debug))

        if fine_hand_gesture == AICameraFineHandGesture.NONE:
            rescue_blocked_by_live_no_hand = (
                fresh_live_results_confirm_no_hand and not effective_hand_boxes
            )

            selected_person_boxes = _prioritize_person_boxes(
                primary_person_box=effective_primary_person_box,
                person_boxes=effective_visible_person_boxes,
                limit=_live_max_person_roi_candidates(self.config),
            )
            debug_snapshot["person_roi_candidate_count"] = len(selected_person_boxes)
            selected_hand_boxes = _prioritize_hand_boxes(
                hand_boxes=effective_hand_boxes,
                limit=_live_max_hand_roi_candidates(self.config),
            )
            debug_snapshot["live_roi_candidate_count"] = len(selected_hand_boxes)
            debug_snapshot["live_roi_hand_box_count"] = len(selected_hand_boxes)

            if (
                selected_person_boxes
                and observe_policy.allow_person_roi_recovery
                and not (
                    observe_policy.prefer_live_hand_roi_before_person_roi
                    and selected_hand_boxes
                )
            ):
                person_roi_choice, person_roi_debug = self._recognize_from_person_rois(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    primary_person_box=effective_primary_person_box,
                    person_boxes=selected_person_boxes,
                    sparse_keypoints=sparse_keypoints,
                    allow_custom=sync_custom_recovery_enabled,
                )
                debug_snapshot.update(person_roi_debug)
                if rescue_blocked_by_live_no_hand and not _allow_person_roi_rescue_despite_live_no_hand(
                    person_roi_choice[0],
                    detection_debug=person_roi_debug.get("person_roi_detection_debug", ()),
                ):
                    debug_snapshot["person_roi_block_reason"] = "fresh_live_results_confirm_no_hand"
                elif person_roi_choice[0] != AICameraFineHandGesture.NONE:
                    fine_hand_gesture, fine_hand_confidence = person_roi_choice
                    debug_snapshot["resolved_source"] = (
                        "recent_visible_person_roi"
                        if len(selected_person_boxes) > 1 and visible_person_box_source == "recent"
                        else "visible_person_roi"
                        if len(selected_person_boxes) > 1
                        else "recent_person_roi"
                        if visible_person_box_source == "recent"
                        else "person_roi"
                    )
            elif selected_person_boxes and not observe_policy.allow_person_roi_recovery:
                debug_snapshot["person_roi_block_reason"] = "observe_policy_disallows_person_roi_recovery"
            elif (
                selected_person_boxes
                and observe_policy.prefer_live_hand_roi_before_person_roi
                and selected_hand_boxes
            ):
                debug_snapshot["person_roi_block_reason"] = "deferred_until_live_hand_roi_exhausted"
            elif rescue_blocked_by_live_no_hand:
                debug_snapshot["person_roi_block_reason"] = "fresh_live_results_confirm_no_hand"

            if fine_hand_gesture == AICameraFineHandGesture.NONE and rescue_blocked_by_live_no_hand:
                debug_snapshot["live_roi_skip_reason"] = "fresh_live_results_confirm_no_hand"
            elif fine_hand_gesture == AICameraFineHandGesture.NONE and observe_policy.allow_live_hand_roi_recovery:
                rescue_choice, live_roi_debug = self._recognize_from_live_hand_rois(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    hand_boxes=selected_hand_boxes,
                    hand_box_source=hand_box_source,
                    allow_custom=sync_custom_recovery_enabled,
                )
                debug_snapshot.update(live_roi_debug)
                if rescue_choice[0] != AICameraFineHandGesture.NONE:
                    fine_hand_gesture, fine_hand_confidence = rescue_choice
                    debug_snapshot["resolved_source"] = (
                        "live_hand_roi" if hand_box_source == "live" else "recent_live_hand_roi"
                    )
            elif fine_hand_gesture == AICameraFineHandGesture.NONE:
                debug_snapshot["live_roi_skip_reason"] = "observe_policy_disallows_live_hand_roi_recovery"

            if (
                fine_hand_gesture == AICameraFineHandGesture.NONE
                and selected_person_boxes
                and observe_policy.allow_person_roi_recovery
                and observe_policy.prefer_live_hand_roi_before_person_roi
                and selected_hand_boxes
            ):
                person_roi_choice, person_roi_debug = self._recognize_from_person_rois(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    primary_person_box=effective_primary_person_box,
                    person_boxes=selected_person_boxes,
                    sparse_keypoints=sparse_keypoints,
                    allow_custom=sync_custom_recovery_enabled,
                )
                debug_snapshot.update(person_roi_debug)
                if rescue_blocked_by_live_no_hand and not _allow_person_roi_rescue_despite_live_no_hand(
                    person_roi_choice[0],
                    detection_debug=person_roi_debug.get("person_roi_detection_debug", ()),
                ):
                    debug_snapshot["person_roi_block_reason"] = "fresh_live_results_confirm_no_hand"
                elif person_roi_choice[0] != AICameraFineHandGesture.NONE:
                    fine_hand_gesture, fine_hand_confidence = person_roi_choice
                    debug_snapshot["resolved_source"] = (
                        "recent_visible_person_roi"
                        if len(selected_person_boxes) > 1 and visible_person_box_source == "recent"
                        else "visible_person_roi"
                        if len(selected_person_boxes) > 1
                        else "recent_person_roi"
                        if visible_person_box_source == "recent"
                        else "person_roi"
                    )

            full_frame_rescue_reason = _resolve_full_frame_rescue_reason(
                fine_hand_gesture=fine_hand_gesture,
                effective_visible_person_boxes=selected_person_boxes,
                effective_hand_boxes=selected_hand_boxes,
                person_roi_detection_count=_coerce_nonnegative_int(
                    debug_snapshot.get("person_roi_detection_count")
                ),
                person_roi_combined_gesture=str(
                    debug_snapshot.get(
                        "person_roi_combined_gesture",
                        AICameraFineHandGesture.NONE.value,
                    )
                    or AICameraFineHandGesture.NONE.value
                ),
                live_roi_hand_box_count=_coerce_nonnegative_int(
                    debug_snapshot.get("live_roi_hand_box_count")
                ),
                live_roi_combined_gesture=str(
                    debug_snapshot.get(
                        "live_roi_combined_gesture",
                        AICameraFineHandGesture.NONE.value,
                    )
                    or AICameraFineHandGesture.NONE.value
                ),
            )
            if (
                rescue_blocked_by_live_no_hand
                and fine_hand_gesture == AICameraFineHandGesture.NONE
                and not _allow_full_frame_rescue_despite_live_no_hand(full_frame_rescue_reason)
            ):
                debug_snapshot["full_frame_hand_attempt_reason"] = "fresh_live_results_confirm_no_hand"
            elif full_frame_rescue_reason is not None:
                if not _full_frame_rescue_allowed_for_reason(
                    observe_policy=observe_policy,
                    reason=full_frame_rescue_reason,
                ):
                    debug_snapshot["full_frame_hand_attempt_reason"] = (
                        "observe_policy_disallows_full_frame_hand_recovery"
                    )
                elif not self._reserve_full_frame_rescue(observed_at):
                    debug_snapshot["full_frame_hand_attempt_reason"] = (
                        f"{full_frame_rescue_reason}_rate_limited"
                    )
                else:
                    full_frame_choice, full_frame_debug = self._recognize_from_full_frame_hand_landmarks(
                        runtime=runtime,
                        frame_rgb=frame_rgb,
                        timestamp_ms=timestamp_ms,
                        allow_custom=sync_custom_recovery_enabled,
                    )
                    debug_snapshot.update(full_frame_debug)
                    debug_snapshot["full_frame_hand_attempt_reason"] = full_frame_rescue_reason
                    hand_count = max(
                        hand_count,
                        _coerce_nonnegative_int(debug_snapshot.get("full_frame_hand_detection_count")),
                    )
                    if full_frame_choice[0] != AICameraFineHandGesture.NONE:
                        fine_hand_gesture, fine_hand_confidence = full_frame_choice
                        debug_snapshot["resolved_source"] = "full_frame_hand_roi"

        if fine_hand_gesture == AICameraFineHandGesture.NONE and deferred_live_stream_choice is not None:
            fine_hand_gesture, fine_hand_confidence = deferred_live_stream_choice
            debug_snapshot["resolved_source"] = "live_stream"
            debug_snapshot["live_stream_recovery_gate"] = (
                "restored_after_sync_custom_recovery_miss"
            )

        raw_resolved_source = str(debug_snapshot.get("resolved_source", "none") or "none")
        stabilized_gesture, stabilized_confidence, temporal_debug = self._stability_tracker.observe(
            observed_at=observed_at,
            gesture=fine_hand_gesture,
            confidence=fine_hand_confidence,
            resolved_source=raw_resolved_source,
            hand_count=hand_count,
            require_weak_live_stream_consensus=observe_policy.require_weak_live_stream_consensus,
        )
        debug_snapshot.update(temporal_debug)
        temporal_resolved_source = str(
            debug_snapshot.get("temporal_resolved_source", raw_resolved_source) or raw_resolved_source
        )
        if temporal_resolved_source != raw_resolved_source:
            debug_snapshot["resolved_source_raw"] = raw_resolved_source
            debug_snapshot["resolved_source"] = temporal_resolved_source
        fine_hand_gesture = stabilized_gesture
        fine_hand_confidence = stabilized_confidence
        authoritative_activation, authoritative_debug = self._authoritative_gesture_lane.observe(
            observed_at=observed_at,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_confidence,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            resolved_source=str(debug_snapshot.get("resolved_source", "none") or "none"),
        )
        debug_snapshot.update(authoritative_debug)

        workflow_decision(
            msg="live_gesture_pipeline_resolution",
            question="Which bounded gesture source should the live gesture pipeline expose for this frame?",
            selected={
                "id": str(debug_snapshot.get("resolved_source", "none") or "none"),
                "summary": "Expose the highest-priority live or rescue gesture source that survived the bounded pipeline.",
            },
            options=[
                {"id": "live_stream", "summary": "Use the direct live-stream recognizer result."},
                {"id": "person_roi", "summary": "Use the primary person ROI hand result."},
                {"id": "visible_person_roi", "summary": "Use a visible-person ROI hand result."},
                {"id": "live_hand_roi", "summary": "Use the tight live hand ROI rescue result."},
                {"id": "full_frame_hand_roi", "summary": "Use the whole-frame hand rescue result."},
                {"id": "temporal_consensus", "summary": "Use the small temporal consensus layer."},
                {"id": "temporal_hold", "summary": "Briefly hold the last stable gesture across a short miss."},
                {"id": "temporal_guard", "summary": "Suppress a weak rescue result until it repeats."},
                {"id": "none", "summary": "Expose no concrete fine-hand gesture from this frame."},
            ],
            context={
                "live_builtin_gesture": builtin_choice[0].value,
                "live_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_custom_gesture": custom_choice[0].value,
                "live_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "person_roi_combined_gesture": debug_snapshot.get("person_roi_combined_gesture"),
                "person_roi_combined_confidence": debug_snapshot.get("person_roi_combined_confidence"),
                "live_roi_combined_gesture": debug_snapshot.get("live_roi_combined_gesture"),
                "live_roi_combined_confidence": debug_snapshot.get("live_roi_combined_confidence"),
                "full_frame_hand_combined_gesture": debug_snapshot.get("full_frame_hand_combined_gesture"),
                "full_frame_hand_combined_confidence": debug_snapshot.get("full_frame_hand_combined_confidence"),
                "temporal_output_gesture": debug_snapshot.get("temporal_output_gesture"),
                "temporal_output_confidence": debug_snapshot.get("temporal_output_confidence"),
                "hand_count": hand_count,
            },
            confidence=_round_optional_confidence(fine_hand_confidence),
            guardrails=["live_gesture_pipeline_priority_order"],
            kpi_impact_estimate={"latency": "medium", "gesture_accuracy": "high"},
        )
        workflow_event(
            kind="metric",
            msg="live_gesture_pipeline_snapshot",
            details={
                "resolved_source": debug_snapshot.get("resolved_source"),
                "hand_count": hand_count,
                "result_age_s": _round_optional_confidence(result_age_s),
            },
        )

        with self._lock:
            self._last_debug_snapshot = dict(debug_snapshot)

        return LiveGestureFrameObservation(
            observed_at=observed_at,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_confidence,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            hand_count=hand_count,
            result_age_s=result_age_s,
            gesture_temporal_authoritative=True,
            gesture_activation_key=authoritative_activation.activation_key,
            gesture_activation_token=authoritative_activation.activation_token,
            gesture_activation_started_at=authoritative_activation.activation_started_at,
            gesture_activation_changed_at=authoritative_activation.activation_changed_at,
            gesture_activation_source=authoritative_activation.activation_source,
            gesture_activation_rising=authoritative_activation.activation_rising,
        )

    def _recognize_from_person_rois(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
        primary_person_box: AICameraBox | None,
        person_boxes: tuple[AICameraBox, ...],
        sparse_keypoints: dict[int, tuple[float, float, float]],
        allow_custom: bool,
    ) -> tuple[_GestureChoice, dict[str, object]]:
        debug: dict[str, object] = {
            "person_roi_candidate_count": len(person_boxes),
            "person_roi_detection_count": 0,
            "person_roi_detection_debug": (),
            "person_roi_match_index": None,
            "person_roi_builtin_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_builtin_confidence": None,
            "person_roi_builtin_detection_index": None,
            "person_roi_custom_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_custom_confidence": None,
            "person_roi_custom_detection_index": None,
            "person_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_combined_confidence": None,
            "person_roi_combined_source": None,
            "person_roi_pose_hint_match_index": None,
            "person_roi_short_circuit_used": False,
            "person_roi_short_circuit_index": None,
            "person_roi_short_circuit_confidence": None,
            "person_roi_short_circuit_reason": None,
        }
        best_builtin_choice = _PersonRoiGestureChoice()
        best_custom_choice = _PersonRoiGestureChoice()
        best_combined_choice = _PersonRoiGestureChoice()
        best_detection_debug: tuple[dict[str, object], ...] = ()
        best_match_index: int | None = None
        pose_hint_match_index: int | None = None
        roi_options: list[dict[str, object]] = []

        for index, person_box in enumerate(person_boxes):
            candidate_sparse_keypoints = self._person_roi_sparse_keypoints(
                person_box=person_box,
                primary_person_box=primary_person_box,
                sparse_keypoints=sparse_keypoints,
            )
            pose_hint_attached = bool(candidate_sparse_keypoints)
            if candidate_sparse_keypoints and pose_hint_match_index is None:
                pose_hint_match_index = index
            try:
                hand_landmark_result = self._ensure_hand_landmark_worker().analyze(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms + index,
                    primary_person_box=person_box,
                    sparse_keypoints=candidate_sparse_keypoints,
                )
            except Exception as exc:
                roi_options.append(
                    {
                        "id": f"person_roi_{index}",
                        "summary": f"ROI {index} hand-landmark stage failed.",
                        "score_components": {
                            "pose_hint_attached": pose_hint_attached,
                            "error": _short_exception(exc),
                        },
                        "constraints_violated": ["hand_landmark_error"],
                    }
                )
                continue

            detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
            debug["person_roi_detection_count"] = (
                _coerce_nonnegative_int(debug.get("person_roi_detection_count")) + len(detections)
            )

            builtin_choice, custom_choice, detection_debug = self._recognize_from_hand_landmark_result(
                runtime=runtime,
                hand_landmark_result=hand_landmark_result,
                allow_custom=allow_custom,
            )
            combined = _combine_person_roi_gesture_choices(
                builtin=builtin_choice,
                custom=custom_choice,
            )

            roi_options.append(
                {
                    "id": f"person_roi_{index}",
                    "summary": f"ROI {index} produced {combined.gesture.value}.",
                    "score_components": {
                        "detection_count": len(detections),
                        "builtin_gesture": builtin_choice.gesture.value,
                        "builtin_confidence": _round_optional_confidence(builtin_choice.confidence),
                        "custom_gesture": custom_choice.gesture.value,
                        "custom_confidence": _round_optional_confidence(custom_choice.confidence),
                        "combined_gesture": combined.gesture.value,
                        "combined_confidence": _round_optional_confidence(combined.confidence),
                        "pose_hint_attached": pose_hint_attached,
                    },
                    "constraints_violated": (
                        [] if combined.gesture != AICameraFineHandGesture.NONE else ["no_symbol_or_source_guard"]
                    ),
                }
            )

            if not best_detection_debug and detection_debug:
                best_detection_debug = detection_debug
            if _prefer_person_roi_gesture_choice(best_combined_choice, combined) != best_combined_choice:
                best_builtin_choice = builtin_choice
                best_custom_choice = custom_choice
                best_combined_choice = combined
                best_detection_debug = detection_debug
                best_match_index = index

            if _should_short_circuit_person_roi_scan(
                config=self.config,
                pose_hint_attached=pose_hint_attached,
                candidate=combined,
            ):
                debug["person_roi_short_circuit_used"] = True
                debug["person_roi_short_circuit_index"] = index
                debug["person_roi_short_circuit_confidence"] = _round_optional_confidence(combined.confidence)
                debug["person_roi_short_circuit_reason"] = _PERSON_ROI_POSE_HINT_SHORT_CIRCUIT_REASON
                break

        debug.update(
            {
                "person_roi_detection_debug": best_detection_debug,
                "person_roi_match_index": best_match_index,
                "person_roi_builtin_gesture": best_builtin_choice.gesture.value,
                "person_roi_builtin_confidence": _round_optional_confidence(best_builtin_choice.confidence),
                "person_roi_builtin_detection_index": best_builtin_choice.detection_index,
                "person_roi_custom_gesture": best_custom_choice.gesture.value,
                "person_roi_custom_confidence": _round_optional_confidence(best_custom_choice.confidence),
                "person_roi_custom_detection_index": best_custom_choice.detection_index,
                "person_roi_combined_gesture": best_combined_choice.gesture.value,
                "person_roi_combined_confidence": _round_optional_confidence(best_combined_choice.confidence),
                "person_roi_combined_source": best_combined_choice.roi_source or None,
                "person_roi_pose_hint_match_index": pose_hint_match_index,
            }
        )

        workflow_decision(
            msg="live_gesture_pipeline_person_roi_selection",
            question="Which visible-person ROI should win the person-conditioned gesture recovery stage?",
            selected={
                "id": "none" if best_match_index is None else f"person_roi_{best_match_index}",
                "summary": "Use the strongest person ROI gesture candidate for this frame.",
            },
            options=roi_options or [{"id": "none", "summary": "No person ROI candidates were available."}],
            context={
                "person_roi_candidate_count": len(person_boxes),
                "person_roi_detection_count": debug["person_roi_detection_count"],
                "pose_hint_match_index": pose_hint_match_index,
                "combined_source": best_combined_choice.roi_source or None,
            },
            confidence=_round_optional_confidence(best_combined_choice.confidence),
            guardrails=["person_roi_priority_order", "person_roi_source_confidence_guard"],
            kpi_impact_estimate={"latency": "medium", "gesture_accuracy": "high"},
        )
        return best_combined_choice.as_choice(), debug

    def _person_roi_sparse_keypoints(
        self,
        *,
        person_box: AICameraBox,
        primary_person_box: AICameraBox | None,
        sparse_keypoints: dict[int, tuple[float, float, float]],
    ) -> dict[int, tuple[float, float, float]]:
        if not sparse_keypoints or primary_person_box is None:
            return {}
        if person_box == primary_person_box:
            return dict(sparse_keypoints)
        try:
            if iou(person_box, primary_person_box) >= _PRIMARY_PERSON_HINT_MIN_IOU:
                return dict(sparse_keypoints)
        except Exception:
            return {}
        return {}

    def _recognize_from_hand_landmark_result(
        self,
        *,
        runtime: dict[str, Any],
        hand_landmark_result: object,
        allow_custom: bool,
    ) -> tuple[_PersonRoiGestureChoice, _PersonRoiGestureChoice, tuple[dict[str, object], ...]]:
        detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
        if not detections:
            return _PersonRoiGestureChoice(), _PersonRoiGestureChoice(), ()

        builtin_choice = _PersonRoiGestureChoice()
        custom_choice = _PersonRoiGestureChoice()
        detection_debug: list[dict[str, object]] = []

        builtin_recognizer = self._runtime.ensure_roi_gesture_recognizer(runtime)
        custom_recognizer = None
        if allow_custom and _live_custom_gesture_enabled(self.config):
            custom_recognizer = self._runtime.ensure_custom_roi_gesture_recognizer(runtime)

        for index, detection in enumerate(detections):
            gesture_frame = _resolve_detection_gesture_frame(detection)
            if gesture_frame is None:
                continue
            roi_source = _coerce_roi_source_value(getattr(detection, "roi_source", None))
            gesture_frame_source = _resolve_detection_gesture_frame_source(detection)
            gesture_context_frame = _resolve_gesture_context_retry_frame(
                primary_frame=gesture_frame,
                context_frame=getattr(detection, "gesture_context_frame_rgb", None),
            )

            (
                builtin_raw_candidate,
                builtin_result,
                builtin_context_retry_used,
                builtin_context_candidate,
                builtin_context_result,
            ) = _recognize_roi_gesture_candidate_with_context_retry(
                runtime_interface=self._runtime,
                runtime=runtime,
                recognizer=builtin_recognizer,
                frame_rgb=gesture_frame,
                context_frame_rgb=gesture_context_frame,
                category_map=_LIVE_BUILTIN_FINE_GESTURE_MAP,
                min_score=self.config.builtin_gesture_min_score,
            )
            builtin_candidate = _guard_person_roi_gesture_choice(
                candidate=builtin_raw_candidate,
                roi_source=roi_source,
                gesture_frame_source=gesture_frame_source,
                detection_index=index,
            )
            builtin_choice = _prefer_person_roi_gesture_choice(builtin_choice, builtin_candidate)

            custom_result: object | None = None
            custom_raw_candidate: _GestureChoice = (AICameraFineHandGesture.NONE, None)
            custom_context_retry_used = False
            custom_context_candidate: _GestureChoice = (AICameraFineHandGesture.NONE, None)
            custom_context_result: object | None = None
            custom_candidate = _PersonRoiGestureChoice(
                roi_source=roi_source,
                detection_index=index,
            )

            if custom_recognizer is not None:
                (
                    custom_raw_candidate,
                    custom_result,
                    custom_context_retry_used,
                    custom_context_candidate,
                    custom_context_result,
                ) = _recognize_roi_gesture_candidate_with_context_retry(
                    runtime_interface=self._runtime,
                    runtime=runtime,
                    recognizer=custom_recognizer,
                    frame_rgb=gesture_frame,
                    context_frame_rgb=gesture_context_frame,
                    category_map=_LIVE_CUSTOM_FINE_GESTURE_MAP,
                    min_score=_live_custom_min_score(self.config),
                )
                custom_candidate = _guard_person_roi_gesture_choice(
                    candidate=custom_raw_candidate,
                    roi_source=roi_source,
                    gesture_frame_source=gesture_frame_source,
                    detection_index=index,
                )
                custom_choice = _prefer_person_roi_gesture_choice(custom_choice, custom_candidate)

            detection_debug.append(
                _summarize_roi_gesture_debug(
                    detection=detection,
                    builtin_raw_candidate=builtin_raw_candidate,
                    builtin_candidate=builtin_candidate,
                    builtin_result=builtin_result,
                    builtin_context_retry_used=builtin_context_retry_used,
                    builtin_context_candidate=builtin_context_candidate,
                    builtin_context_result=builtin_context_result,
                    custom_raw_candidate=custom_raw_candidate,
                    custom_candidate=custom_candidate,
                    custom_result=custom_result,
                    custom_context_retry_used=custom_context_retry_used,
                    custom_context_candidate=custom_context_candidate,
                    custom_context_result=custom_context_result,
                )
            )

        return builtin_choice, custom_choice, tuple(detection_debug)

    def _make_builtin_result_handler(self, generation: int):
        def _callback(result: object, output_image: object, timestamp_ms: int) -> None:
            self._handle_builtin_result(
                result=result,
                _output_image=output_image,
                timestamp_ms=timestamp_ms,
                generation=generation,
            )
        setattr(_callback, MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR, self._builtin_callback_cache_key)
        return _callback

    def _make_custom_result_handler(self, generation: int):
        def _callback(result: object, output_image: object, timestamp_ms: int) -> None:
            self._handle_custom_result(
                result=result,
                _output_image=output_image,
                timestamp_ms=timestamp_ms,
                generation=generation,
            )
        setattr(_callback, MEDIAPIPE_LIVE_CALLBACK_CACHE_KEY_ATTR, self._custom_callback_cache_key)
        return _callback

    def _handle_builtin_result(
        self,
        *,
        result: object,
        _output_image: object,
        timestamp_ms: int,
        generation: int,
    ) -> None:
        gesture, confidence = resolve_fine_hand_gesture(
            result=result,
            category_map=_LIVE_BUILTIN_FINE_GESTURE_MAP,
            min_score=self.config.builtin_gesture_min_score,
        )
        snapshot = _RecognizerSnapshot(
            timestamp_ms=max(1, int(timestamp_ms)),
            gesture=gesture,
            confidence=confidence,
            hand_count=_count_hand_landmarks(result),
            open_palm_center_x=_extract_open_palm_center_x(
                result,
                min_score=self.config.builtin_gesture_min_score,
            ),
            hand_boxes=_extract_hand_boxes(result),
        )
        with self._result_condition:
            if generation != self._callback_generation:
                return
            if (
                self._pending_live_builtin_timestamp_ms is not None
                and timestamp_ms >= self._pending_live_builtin_timestamp_ms
            ):
                self._pending_live_builtin_timestamp_ms = None
                self._pending_live_builtin_submitted_mono_s = None
            _remember_snapshot(
                self._builtin_history,
                snapshot,
                max_items=_live_result_history_size(self.config),
            )
            if self._latest_builtin is None or snapshot.timestamp_ms >= self._latest_builtin.timestamp_ms:
                self._latest_builtin = snapshot
            self._result_condition.notify_all()

    def _handle_custom_result(
        self,
        *,
        result: object,
        _output_image: object,
        timestamp_ms: int,
        generation: int,
    ) -> None:
        gesture, confidence = resolve_fine_hand_gesture(
            result=result,
            category_map=_LIVE_CUSTOM_FINE_GESTURE_MAP,
            min_score=_live_custom_min_score(self.config),
        )
        snapshot = _RecognizerSnapshot(
            timestamp_ms=max(1, int(timestamp_ms)),
            gesture=gesture,
            confidence=confidence,
            hand_count=_count_hand_landmarks(result),
            open_palm_center_x=None,
            hand_boxes=_extract_hand_boxes(result),
        )
        with self._result_condition:
            if generation != self._callback_generation:
                return
            if (
                self._pending_live_custom_timestamp_ms is not None
                and timestamp_ms >= self._pending_live_custom_timestamp_ms
            ):
                self._pending_live_custom_timestamp_ms = None
                self._pending_live_custom_submitted_mono_s = None
            _remember_snapshot(
                self._custom_history,
                snapshot,
                max_items=_live_result_history_size(self.config),
            )
            if self._latest_custom is None or snapshot.timestamp_ms >= self._latest_custom.timestamp_ms:
                self._latest_custom = snapshot
            self._result_condition.notify_all()

    def _recognize_from_live_hand_rois(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        hand_boxes: tuple[_HandBox, ...],
        hand_box_source: str,
        allow_custom: bool,
    ) -> tuple[_GestureChoice, dict[str, object]]:
        debug: dict[str, object] = {
            "live_roi_hand_box_count": len(hand_boxes),
            "live_roi_hand_box_source": hand_box_source,
            "live_roi_detection_debug": (),
            "live_roi_builtin_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_builtin_confidence": None,
            "live_roi_custom_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_custom_confidence": None,
            "live_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_combined_confidence": None,
        }
        if not hand_boxes:
            return (AICameraFineHandGesture.NONE, None), debug

        builtin_recognizer = self._runtime.ensure_roi_gesture_recognizer(runtime)
        custom_recognizer = None
        if allow_custom and _live_custom_gesture_enabled(self.config):
            custom_recognizer = self._runtime.ensure_custom_roi_gesture_recognizer(runtime)

        builtin_choice: _GestureChoice = (AICameraFineHandGesture.NONE, None)
        custom_choice: _GestureChoice = (AICameraFineHandGesture.NONE, None)
        hand_box_options: list[dict[str, object]] = []
        hand_box_debug: list[dict[str, object]] = []

        for index, hand_box in enumerate(hand_boxes):
            crop = _crop_hand_box(frame_rgb, hand_box)
            if crop is None:
                hand_box_options.append(
                    {
                        "id": f"live_hand_roi_{index}",
                        "summary": f"Hand ROI {index} crop failed.",
                        "score_components": {"hand_box_index": index},
                        "constraints_violated": ["crop_failed"],
                    }
                )
                continue
            try:
                image = self._runtime.build_image(runtime, frame_rgb=crop)
                builtin_result = builtin_recognizer.recognize(image)
            except Exception as exc:
                hand_box_debug.append({"hand_box_index": index, "error": _short_exception(exc)})
                hand_box_options.append(
                    {
                        "id": f"live_hand_roi_{index}",
                        "summary": f"Hand ROI {index} recognizer failed.",
                        "score_components": {"error": _short_exception(exc)},
                        "constraints_violated": ["recognize_error"],
                    }
                )
                continue

            builtin_candidate = resolve_fine_hand_gesture(
                result=builtin_result,
                category_map=_LIVE_BUILTIN_FINE_GESTURE_MAP,
                min_score=self.config.builtin_gesture_min_score,
            )
            builtin_choice = prefer_gesture_choice(builtin_choice, builtin_candidate)

            custom_result = None
            custom_candidate: _GestureChoice = (AICameraFineHandGesture.NONE, None)
            if custom_recognizer is not None:
                try:
                    custom_result = custom_recognizer.recognize(image)
                    custom_candidate = resolve_fine_hand_gesture(
                        result=custom_result,
                        category_map=_LIVE_CUSTOM_FINE_GESTURE_MAP,
                        min_score=_live_custom_min_score(self.config),
                    )
                    custom_choice = prefer_gesture_choice(custom_choice, custom_candidate)
                except Exception as exc:
                    hand_box_debug.append(
                        {
                            "hand_box_index": index,
                            "custom_error": _short_exception(exc),
                            "builtin_gesture": builtin_candidate[0].value,
                            "builtin_confidence": _round_optional_confidence(builtin_candidate[1]),
                        }
                    )
                    custom_result = None

            combined_candidate = combine_task_specific_custom_gesture_choice(
                builtin_candidate,
                custom_candidate,
                preferred_custom_gestures=_LIVE_FINE_GESTURE_ALLOWLIST,
            )
            hand_box_debug.append(
                {
                    "hand_box_index": index,
                    "builtin_gesture": builtin_candidate[0].value,
                    "builtin_confidence": _round_optional_confidence(builtin_candidate[1]),
                    "builtin_categories": _summarize_gesture_categories(builtin_result),
                    "custom_gesture": custom_candidate[0].value,
                    "custom_confidence": _round_optional_confidence(custom_candidate[1]),
                    "custom_categories": _summarize_gesture_categories(custom_result),
                    "combined_gesture": combined_candidate[0].value,
                    "combined_confidence": _round_optional_confidence(combined_candidate[1]),
                }
            )
            hand_box_options.append(
                {
                    "id": f"live_hand_roi_{index}",
                    "summary": f"Hand ROI {index} produced {combined_candidate[0].value}.",
                    "score_components": {
                        "builtin_gesture": builtin_candidate[0].value,
                        "builtin_confidence": _round_optional_confidence(builtin_candidate[1]),
                        "custom_gesture": custom_candidate[0].value,
                        "custom_confidence": _round_optional_confidence(custom_candidate[1]),
                        "combined_gesture": combined_candidate[0].value,
                        "combined_confidence": _round_optional_confidence(combined_candidate[1]),
                    },
                    "constraints_violated": (
                        [] if combined_candidate[0] != AICameraFineHandGesture.NONE else ["no_symbol"]
                    ),
                }
            )

        combined = combine_task_specific_custom_gesture_choice(
            builtin_choice,
            custom_choice,
            preferred_custom_gestures=_LIVE_FINE_GESTURE_ALLOWLIST,
        )
        debug.update(
            {
                "live_roi_detection_debug": tuple(hand_box_debug),
                "live_roi_builtin_gesture": builtin_choice[0].value,
                "live_roi_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_roi_custom_gesture": custom_choice[0].value,
                "live_roi_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "live_roi_combined_gesture": combined[0].value,
                "live_roi_combined_confidence": _round_optional_confidence(combined[1]),
            }
        )

        workflow_decision(
            msg="live_gesture_pipeline_live_hand_roi_selection",
            question="Which tight live hand ROI should win the hand-box rescue stage?",
            selected={
                "id": combined[0].value,
                "summary": "Use the strongest tight live hand ROI rescue candidate for this frame.",
            },
            options=hand_box_options or [{"id": "none", "summary": "No live hand ROI boxes were available."}],
            context={
                "hand_box_count": len(hand_boxes),
                "hand_box_source": hand_box_source,
            },
            confidence=_round_optional_confidence(combined[1]),
            guardrails=["live_hand_roi_priority_order"],
            kpi_impact_estimate={"latency": "medium", "gesture_accuracy": "medium"},
        )
        return combined, debug

    def _recognize_from_full_frame_hand_landmarks(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        timestamp_ms: int,
        allow_custom: bool,
    ) -> tuple[_GestureChoice, dict[str, object]]:
        debug: dict[str, object] = {
            "full_frame_hand_detection_count": 0,
            "full_frame_hand_detection_debug": (),
            "full_frame_hand_builtin_gesture": AICameraFineHandGesture.NONE.value,
            "full_frame_hand_builtin_confidence": None,
            "full_frame_hand_custom_gesture": AICameraFineHandGesture.NONE.value,
            "full_frame_hand_custom_confidence": None,
            "full_frame_hand_combined_gesture": AICameraFineHandGesture.NONE.value,
            "full_frame_hand_combined_confidence": None,
        }
        try:
            hand_landmark_result = self._ensure_hand_landmark_worker().analyze_full_frame(
                runtime=runtime,
                frame_rgb=frame_rgb,
                timestamp_ms=timestamp_ms,
            )
        except Exception as exc:
            debug["full_frame_hand_detection_debug"] = (
                {"error": _short_exception(exc)},
            )
            return (AICameraFineHandGesture.NONE, None), debug

        detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
        debug["full_frame_hand_detection_count"] = len(detections)

        builtin_choice, custom_choice, detection_debug = self._recognize_from_hand_landmark_result(
            runtime=runtime,
            hand_landmark_result=hand_landmark_result,
            allow_custom=allow_custom,
        )
        combined = combine_task_specific_custom_gesture_choice(
            builtin_choice.as_choice(),
            custom_choice.as_choice(),
            preferred_custom_gestures=_LIVE_FINE_GESTURE_ALLOWLIST,
        )
        debug.update(
            {
                "full_frame_hand_detection_debug": detection_debug,
                "full_frame_hand_builtin_gesture": builtin_choice.gesture.value,
                "full_frame_hand_builtin_confidence": _round_optional_confidence(builtin_choice.confidence),
                "full_frame_hand_builtin_detection_index": builtin_choice.detection_index,
                "full_frame_hand_custom_gesture": custom_choice.gesture.value,
                "full_frame_hand_custom_confidence": _round_optional_confidence(custom_choice.confidence),
                "full_frame_hand_custom_detection_index": custom_choice.detection_index,
                "full_frame_hand_combined_gesture": combined[0].value,
                "full_frame_hand_combined_confidence": _round_optional_confidence(combined[1]),
            }
        )
        workflow_decision(
            msg="live_gesture_pipeline_full_frame_selection",
            question="Should the final whole-frame hand rescue contribute a concrete gesture?",
            selected={
                "id": combined[0].value,
                "summary": "Use the strongest whole-frame hand rescue candidate for this frame.",
            },
            options=[
                {"id": "concrete_gesture", "summary": "Accept the whole-frame hand rescue as the winning source."},
                {"id": "none", "summary": "Reject the whole-frame hand rescue because it found no concrete gesture."},
            ],
            context={
                "detection_count": len(detections),
                "builtin_detection_index": builtin_choice.detection_index,
                "custom_detection_index": custom_choice.detection_index,
            },
            confidence=_round_optional_confidence(combined[1]),
            guardrails=["full_frame_hand_rescue"],
            kpi_impact_estimate={"latency": "high", "gesture_accuracy": "fallback_only"},
        )
        return combined, debug

    def _fresh_recent_primary_person_box(self, observed_at: float) -> AICameraBox | None:
        if self._recent_primary_person_box is None or self._recent_primary_person_seen_at is None:
            return None
        if (observed_at - self._recent_primary_person_seen_at) > _RECENT_PRIMARY_PERSON_BOX_TTL_S:
            return None
        return self._recent_primary_person_box

    def _fresh_recent_visible_person_boxes(self, observed_at: float) -> tuple[AICameraBox, ...]:
        if not self._recent_visible_person_boxes or self._recent_visible_person_boxes_seen_at is None:
            return ()
        if (observed_at - self._recent_visible_person_boxes_seen_at) > _RECENT_VISIBLE_PERSON_BOXES_TTL_S:
            return ()
        return tuple(self._recent_visible_person_boxes)

    def _fresh_recent_hand_boxes(self, observed_at: float) -> tuple[_HandBox, ...]:
        if not self._recent_hand_boxes or self._recent_hand_boxes_seen_at is None:
            return ()
        if (observed_at - self._recent_hand_boxes_seen_at) > _RECENT_HAND_BOXES_TTL_S:
            return ()
        return tuple(self._recent_hand_boxes)

    def _ensure_hand_landmark_worker(self) -> MediaPipeHandLandmarkWorker:
        with self._lock:
            if self._hand_landmark_worker is not None:
                return self._hand_landmark_worker
            self._hand_landmark_worker = MediaPipeHandLandmarkWorker(
                config=HandLandmarkWorkerConfig.from_config(self.config),
            )
            return self._hand_landmark_worker

    def _submit_live_builtin_frame(
        self,
        *,
        runtime: dict[str, Any],
        image: Any,
        timestamp_ms: int,
        generation: int,
        observed_mono_s: float,
    ) -> bool:
        with self._lock:
            if self._pending_live_builtin_timestamp_ms is not None:
                return False
            self._pending_live_builtin_timestamp_ms = timestamp_ms
            self._pending_live_builtin_submitted_mono_s = observed_mono_s
        try:
            builtin_recognizer = self._runtime.ensure_live_gesture_recognizer(
                runtime,
                result_callback=self._make_builtin_result_handler(generation),
                num_hands_override=_LIVE_GESTURE_NUM_HANDS,
            )
            builtin_recognizer.recognize_async(image, timestamp_ms)
            return True
        except Exception:
            with self._lock:
                if self._pending_live_builtin_timestamp_ms == timestamp_ms:
                    self._pending_live_builtin_timestamp_ms = None
                    self._pending_live_builtin_submitted_mono_s = None
            raise

    def _submit_live_custom_frame(
        self,
        *,
        runtime: dict[str, Any],
        image: Any,
        timestamp_ms: int,
        generation: int,
        observed_mono_s: float,
    ) -> bool:
        with self._lock:
            if self._pending_live_custom_timestamp_ms is not None:
                return False
            self._pending_live_custom_timestamp_ms = timestamp_ms
            self._pending_live_custom_submitted_mono_s = observed_mono_s
        try:
            custom_recognizer = self._runtime.ensure_live_custom_gesture_recognizer(
                runtime,
                result_callback=self._make_custom_result_handler(generation),
            )
            custom_recognizer.recognize_async(image, timestamp_ms)
            return True
        except Exception:
            with self._lock:
                if self._pending_live_custom_timestamp_ms == timestamp_ms:
                    self._pending_live_custom_timestamp_ms = None
                    self._pending_live_custom_submitted_mono_s = None
            raise

    def _reset_stale_live_submissions(
        self,
        *,
        observed_mono_s: float,
    ) -> dict[str, object]:
        timeout_s = _live_pending_result_timeout_s(self.config)
        debug: dict[str, object] = {
            "live_pending_backpressure": False,
            "live_pending_reset": False,
            "live_pending_reset_reason": None,
            "live_builtin_pending_age_s": None,
            "live_custom_pending_age_s": None,
        }

        reset_required = False
        with self._lock:
            builtin_age_s = (
                None
                if self._pending_live_builtin_submitted_mono_s is None
                else max(0.0, observed_mono_s - self._pending_live_builtin_submitted_mono_s)
            )
            custom_age_s = (
                None
                if self._pending_live_custom_submitted_mono_s is None
                else max(0.0, observed_mono_s - self._pending_live_custom_submitted_mono_s)
            )
            debug["live_builtin_pending_age_s"] = _round_optional_confidence(builtin_age_s)
            debug["live_custom_pending_age_s"] = _round_optional_confidence(custom_age_s)
            debug["live_pending_backpressure"] = (
                self._pending_live_builtin_timestamp_ms is not None
                or self._pending_live_custom_timestamp_ms is not None
            )
            builtin_stale = builtin_age_s is not None and builtin_age_s >= timeout_s
            custom_stale = custom_age_s is not None and custom_age_s >= timeout_s
            if builtin_stale or custom_stale:
                self._callback_generation += 1
                self._latest_builtin = None
                self._latest_custom = None
                self._builtin_history.clear()
                self._custom_history.clear()
                self._pending_live_builtin_timestamp_ms = None
                self._pending_live_custom_timestamp_ms = None
                self._pending_live_builtin_submitted_mono_s = None
                self._pending_live_custom_submitted_mono_s = None
                debug["live_pending_backpressure"] = False
                debug["live_pending_reset"] = True
                reasons: list[str] = []
                if builtin_stale:
                    reasons.append("builtin_timeout")
                if custom_stale:
                    reasons.append("custom_timeout")
                debug["live_pending_reset_reason"] = "+".join(reasons)
                reset_required = True
        if reset_required:
            self._runtime.reset_live_gesture_recognizers()
        return debug

    def _select_snapshot_from_history(
        self,
        *,
        history: OrderedDict[int, _RecognizerSnapshot],
        max_timestamp_ms: int,
        frame_time_s: float,
    ) -> _RecognizerSnapshot | None:
        for ts, snapshot in reversed(tuple(history.items())):
            if ts > max_timestamp_ms:
                continue
            if _fresh_snapshot(snapshot, frame_time_s=frame_time_s) is not None:
                return snapshot
        return None

    def _reserve_full_frame_rescue(self, observed_at: float) -> bool:
        min_interval_s = _live_full_frame_rescue_min_interval_s(self.config)
        with self._lock:
            if (
                self._last_full_frame_rescue_at is not None
                and (observed_at - self._last_full_frame_rescue_at) < min_interval_s
            ):
                return False
            self._last_full_frame_rescue_at = observed_at
            return True

    def _next_live_timestamp_ms(self, observed_at: float) -> int:
        proposed = max(1, int(self._runtime.timestamp_ms(observed_at)))
        with self._lock:
            if proposed <= self._last_submitted_live_timestamp_ms:
                proposed = self._last_submitted_live_timestamp_ms + 1
            self._last_submitted_live_timestamp_ms = proposed
            return proposed

    def _await_current_live_results(
        self,
        *,
        timestamp_ms: int,
        expect_builtin: bool,
        expect_custom: bool,
    ) -> tuple[float, bool, bool]:
        wait_started_at = monotonic()
        if not expect_builtin and not expect_custom:
            return 0.0, False, False
        with self._result_condition:
            builtin_ready = (not expect_builtin) or (timestamp_ms in self._builtin_history)
            custom_ready = (not expect_custom) or (timestamp_ms in self._custom_history)
            if not (builtin_ready and custom_ready):
                self._result_condition.wait_for(
                    lambda: (
                        ((not expect_builtin) or (timestamp_ms in self._builtin_history))
                        and ((not expect_custom) or (timestamp_ms in self._custom_history))
                    ),
                    timeout=_DEFAULT_CURRENT_RESULT_WAIT_S,
                )
                builtin_ready = (not expect_builtin) or (timestamp_ms in self._builtin_history)
                custom_ready = (not expect_custom) or (timestamp_ms in self._custom_history)
        return monotonic() - wait_started_at, (builtin_ready and expect_builtin), (custom_ready and expect_custom)

    def _fail_closed_observation(
        self,
        *,
        observed_at: float,
        observe_policy: LiveGestureObservePolicy,
        person_count: int,
        visible_person_boxes: tuple[AICameraBox, ...],
        sparse_keypoints: dict[int, tuple[float, float, float]],
        stage: str,
        exc: Exception,
    ) -> LiveGestureFrameObservation:
        debug_snapshot = {
            "resolved_source": "none",
            "observe_policy": observe_policy.name,
            "input_person_count": max(0, int(person_count)),
            "visible_person_box_count": len(tuple(visible_person_boxes or ())),
            "pose_hint_keypoint_count": len(sparse_keypoints),
            "error_stage": stage,
            "error_type": type(exc).__name__,
            "error": _short_exception(exc),
            "fail_closed": True,
        }
        workflow_event(
            kind="error",
            msg="live_gesture_pipeline_error",
            details=dict(debug_snapshot),
        )
        with self._lock:
            self._last_debug_snapshot = dict(debug_snapshot)
        return LiveGestureFrameObservation(observed_at=observed_at)


def _fresh_snapshot(
    snapshot: _RecognizerSnapshot | None,
    *,
    frame_time_s: float,
) -> _RecognizerSnapshot | None:
    if snapshot is None:
        return None
    age_s = frame_time_s - (snapshot.timestamp_ms / 1000.0)
    if not math.isfinite(age_s) or age_s < 0.0 or age_s > _DEFAULT_MAX_RESULT_AGE_S:
        return None
    return snapshot


def _remember_snapshot(
    history: OrderedDict[int, _RecognizerSnapshot],
    snapshot: _RecognizerSnapshot,
    *,
    max_items: int,
) -> None:
    history[snapshot.timestamp_ms] = snapshot
    while len(history) > max_items:
        history.popitem(last=False)


def _count_hand_landmarks(result: object) -> int:
    hand_landmarks = getattr(result, "hand_landmarks", None)
    if hand_landmarks is None:
        return 0
    try:
        return max(0, len(hand_landmarks))
    except TypeError:
        return 0


def _extract_open_palm_center_x(
    result: object,
    *,
    min_score: float,
) -> float | None:
    gestures = getattr(result, "gestures", None)
    hand_landmarks = getattr(result, "hand_landmarks", None)
    if gestures is None or hand_landmarks is None:
        return None
    try:
        gesture_sets = list(gestures)
        landmark_sets = list(hand_landmarks)
    except TypeError:
        return None

    best_score = 0.0
    best_center_x = None
    for index, categories in enumerate(gesture_sets):
        if index >= len(landmark_sets):
            break
        label_score = _open_palm_score(categories)
        if label_score is None or label_score < min_score or label_score <= best_score:
            continue
        center_x = _hand_center_x(landmark_sets[index])
        if center_x is None:
            continue
        best_score = label_score
        best_center_x = center_x
    return best_center_x


def _extract_hand_boxes(result: object) -> tuple[_HandBox, ...]:
    hand_landmarks = getattr(result, "hand_landmarks", None)
    if hand_landmarks is None:
        return ()
    try:
        landmark_sets = list(hand_landmarks)
    except TypeError:
        return ()

    boxes: list[_HandBox] = []
    for landmarks in landmark_sets:
        try:
            points = list(landmarks)
        except TypeError:
            continue

        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            x = _coerce_unit_interval(getattr(point, "x", None))
            y = _coerce_unit_interval(getattr(point, "y", None))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        if not xs or not ys:
            continue
        left = max(0.0, min(xs) - 0.08)
        right = min(1.0, max(xs) + 0.08)
        top = max(0.0, min(ys) - 0.08)
        bottom = min(1.0, max(ys) + 0.08)
        if right <= left or bottom <= top:
            continue
        boxes.append((top, left, bottom, right))
    return tuple(boxes)


def _crop_hand_box(
    frame_rgb: Any,
    hand_box: _HandBox,
) -> Any | None:
    if getattr(frame_rgb, "shape", None) is None:
        return frame_rgb
    try:
        frame_height = int(frame_rgb.shape[0])
        frame_width = int(frame_rgb.shape[1])
    except Exception:
        return None
    if frame_height <= 1 or frame_width <= 1:
        return None

    top, left, bottom, right = hand_box
    crop_width = max(0.0, right - left)
    crop_height = max(0.0, bottom - top)
    if crop_width <= 0.0 or crop_height <= 0.0:
        return None

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    side = max(
        crop_width + (_LIVE_HAND_CROP_PADDING * 2.0),
        crop_height + (_LIVE_HAND_CROP_PADDING * 2.0),
        _LIVE_HAND_MIN_CONTEXT_RATIO,
    )
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

    y0 = max(0, min(frame_height - 1, int(math.floor(top * frame_height))))
    x0 = max(0, min(frame_width - 1, int(math.floor(left * frame_width))))
    y1 = max(y0 + 1, min(frame_height, int(math.ceil(bottom * frame_height))))
    x1 = max(x0 + 1, min(frame_width, int(math.ceil(right * frame_width))))
    try:
        return frame_rgb[y0:y1, x0:x1]
    except Exception:
        return None


def _resolve_gesture_context_retry_frame(
    *,
    primary_frame: Any,
    context_frame: Any,
) -> Any | None:
    if context_frame is None or context_frame is primary_frame:
        return None
    primary_shape = getattr(primary_frame, "shape", None)
    context_shape = getattr(context_frame, "shape", None)
    if primary_shape is not None and context_shape is not None:
        try:
            if tuple(primary_shape) == tuple(context_shape):
                return None
        except TypeError:
            return context_frame
        return context_frame
    if primary_shape is None and context_shape is None:
        try:
            if primary_frame == context_frame:
                return None
        except Exception:
            return context_frame
    return context_frame


def _recognize_roi_gesture_candidate_with_context_retry(
    *,
    runtime_interface: Any,
    runtime: dict[str, Any],
    recognizer: Any,
    frame_rgb: Any,
    context_frame_rgb: Any | None,
    category_map: dict[str, AICameraFineHandGesture],
    min_score: float,
) -> tuple[_GestureChoice, object | None, bool, _GestureChoice, object | None]:
    try:
        image = runtime_interface.build_image(runtime, frame_rgb=frame_rgb)
        primary_result = recognizer.recognize(image)
    except Exception:
        return (AICameraFineHandGesture.NONE, None), None, False, (AICameraFineHandGesture.NONE, None), None

    primary_candidate = resolve_fine_hand_gesture(
        result=primary_result,
        category_map=category_map,
        min_score=min_score,
    )
    if primary_candidate[0] != AICameraFineHandGesture.NONE or context_frame_rgb is None:
        return primary_candidate, primary_result, False, (AICameraFineHandGesture.NONE, None), None

    try:
        context_image = runtime_interface.build_image(runtime, frame_rgb=context_frame_rgb)
        context_result = recognizer.recognize(context_image)
    except Exception:
        return primary_candidate, primary_result, True, (AICameraFineHandGesture.NONE, None), None

    context_candidate = resolve_fine_hand_gesture(
        result=context_result,
        category_map=category_map,
        min_score=min_score,
    )
    if context_candidate[0] != AICameraFineHandGesture.NONE:
        return context_candidate, context_result, True, context_candidate, context_result
    return primary_candidate, primary_result, True, context_candidate, context_result


def _summarize_roi_gesture_debug(
    *,
    detection: object,
    builtin_raw_candidate: _GestureChoice,
    builtin_candidate: _PersonRoiGestureChoice,
    builtin_result: object | None,
    builtin_context_retry_used: bool,
    builtin_context_candidate: _GestureChoice,
    builtin_context_result: object | None,
    custom_raw_candidate: _GestureChoice,
    custom_candidate: _PersonRoiGestureChoice,
    custom_result: object | None,
    custom_context_retry_used: bool,
    custom_context_candidate: _GestureChoice,
    custom_context_result: object | None,
) -> dict[str, object]:
    roi_source_value = _coerce_roi_source_value(getattr(detection, "roi_source", None))
    gesture_frame_source = _resolve_detection_gesture_frame_source(detection)
    source_min_confidence = _effective_person_roi_source_min_confidence(
        roi_source=roi_source_value,
        gesture_frame_source=gesture_frame_source,
    )
    gesture_frame = _resolve_detection_gesture_frame(detection)
    return {
        "roi_source": roi_source_value,
        "handedness": getattr(detection, "handedness", None),
        "handedness_score": _round_optional_confidence(getattr(detection, "handedness_score", None)),
        "detection_confidence": _round_optional_confidence(getattr(detection, "confidence", None)),
        "gesture_frame_source": gesture_frame_source,
        "roi_frame_shape": _summarize_frame_shape(getattr(detection, "roi_frame_rgb", None)),
        "gesture_frame_shape": _summarize_frame_shape(gesture_frame),
        "gesture_context_frame_shape": _summarize_frame_shape(
            getattr(detection, "gesture_context_frame_rgb", None)
        ),
        "roi_source_priority": _person_roi_source_priority(roi_source_value),
        "builtin_raw_gesture": builtin_raw_candidate[0].value,
        "builtin_raw_confidence": _round_optional_confidence(builtin_raw_candidate[1]),
        "builtin_context_retry_used": builtin_context_retry_used,
        "builtin_context_gesture": builtin_context_candidate[0].value,
        "builtin_context_confidence": _round_optional_confidence(builtin_context_candidate[1]),
        "builtin_context_categories": _summarize_gesture_categories(builtin_context_result),
        "builtin_gesture": builtin_candidate.gesture.value,
        "builtin_confidence": _round_optional_confidence(builtin_candidate.confidence),
        "builtin_source_min_confidence": _round_optional_confidence(source_min_confidence),
        "builtin_source_accepted": builtin_candidate.gesture != AICameraFineHandGesture.NONE,
        "builtin_categories": _summarize_gesture_categories(builtin_result),
        "custom_raw_gesture": custom_raw_candidate[0].value,
        "custom_raw_confidence": _round_optional_confidence(custom_raw_candidate[1]),
        "custom_context_retry_used": custom_context_retry_used,
        "custom_context_gesture": custom_context_candidate[0].value,
        "custom_context_confidence": _round_optional_confidence(custom_context_candidate[1]),
        "custom_context_categories": _summarize_gesture_categories(custom_context_result),
        "custom_gesture": custom_candidate.gesture.value,
        "custom_confidence": _round_optional_confidence(custom_candidate.confidence),
        "custom_source_min_confidence": _round_optional_confidence(source_min_confidence),
        "custom_source_accepted": custom_candidate.gesture != AICameraFineHandGesture.NONE,
        "custom_categories": _summarize_gesture_categories(custom_result),
    }


def _resolve_detection_gesture_frame(detection: object) -> object | None:
    gesture_frame = getattr(detection, "gesture_frame_rgb", None)
    if gesture_frame is not None:
        return gesture_frame
    return getattr(detection, "roi_frame_rgb", None)


def _resolve_detection_gesture_frame_source(detection: object) -> str:
    if getattr(detection, "gesture_frame_rgb", None) is not None:
        return "full_frame_landmark_crop"
    return "roi_local_crop"


def _live_custom_gesture_enabled(config: MediaPipeVisionConfig) -> bool:
    return bool(getattr(config, "custom_gesture_model_path", None)) and bool(_LIVE_CUSTOM_FINE_GESTURE_MAP)


def _live_custom_min_score(config: MediaPipeVisionConfig) -> float:
    configured = _coerce_unit_interval(getattr(config, "custom_gesture_min_score", None))
    return max(_LIVE_CUSTOM_MIN_SCORE_FLOOR, configured or 0.0)


def _fresh_live_results_confirm_no_hand(
    *,
    builtin_ready: bool,
    custom_ready: bool | None,
    hand_count: int,
    live_hand_box_count: int,
) -> bool:
    if not builtin_ready:
        return False
    if custom_ready is None:
        return False
    return hand_count <= 0 and live_hand_box_count <= 0


def _resolve_sync_custom_recovery_mode(
    *,
    observe_policy: LiveGestureObservePolicy,
    live_custom_enabled: bool,
    current_live_custom_ready: bool | None,
) -> tuple[bool, str]:
    if observe_policy.enable_custom_roi:
        return True, "policy_enabled"
    if not observe_policy.allow_custom_recovery_when_live_custom_pending:
        return False, "observe_policy_disabled"
    if not live_custom_enabled:
        return False, "live_custom_disabled"
    if current_live_custom_ready is False:
        return True, "live_custom_pending"
    if current_live_custom_ready:
        return False, "current_live_custom_ready"
    return False, "live_custom_status_unknown"


def _should_defer_live_stream_candidate_for_sync_custom_recovery(
    *,
    config: MediaPipeVisionConfig,
    observe_policy: LiveGestureObservePolicy,
    sync_custom_recovery_enabled: bool,
    gesture: AICameraFineHandGesture,
    confidence: float | None,
) -> bool:
    if gesture == AICameraFineHandGesture.NONE:
        return False
    if not sync_custom_recovery_enabled:
        return False
    if not observe_policy.require_weak_live_stream_consensus:
        return False
    score = _coerce_unit_interval(confidence) or 0.0
    return score < _live_temporal_strong_confidence(config)


def _allow_person_roi_rescue_despite_live_no_hand(
    gesture: AICameraFineHandGesture,
    *,
    detection_debug: object = (),
) -> bool:
    if gesture in _LIVE_NO_HAND_PERSON_ROI_ALLOWLIST:
        return True
    if gesture == AICameraFineHandGesture.NONE:
        return False
    try:
        detections = tuple(cast(Any, detection_debug) or ())
    except TypeError:
        return False
    return any(
        isinstance(entry, Mapping)
        and entry.get("gesture_frame_source") == "full_frame_landmark_crop"
        and (
            bool(entry.get("builtin_source_accepted"))
            or bool(entry.get("custom_source_accepted"))
        )
        for entry in detections
    )


def _allow_full_frame_rescue_despite_live_no_hand(reason: str | None) -> bool:
    return reason == "visible_person_roi_with_hand_detection_without_symbol"


def _summarize_gesture_categories(
    result: object | None,
    *,
    limit: int = 3,
) -> tuple[dict[str, object], ...]:
    if result is None or limit <= 0:
        return ()
    try:
        gesture_sets = tuple(getattr(result, "gestures", None) or ())
    except TypeError:
        return ()
    if not gesture_sets:
        return ()
    summary: list[dict[str, object]] = []
    for category in tuple(gesture_sets[0] or ())[:limit]:
        raw_label = str(getattr(category, "category_name", "") or "").strip()
        summary.append(
            {
                "label": raw_label,
                "normalized_label": normalize_category_name(raw_label),
                "score": _round_optional_confidence(getattr(category, "score", None)),
            }
        )
    return tuple(summary)


def _summarize_frame_shape(frame_rgb: object | None) -> tuple[int, ...] | None:
    if frame_rgb is None:
        return None
    shape = getattr(frame_rgb, "shape", None)
    if not shape:
        return None
    try:
        return tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        return None


def _resolve_full_frame_rescue_reason(
    *,
    fine_hand_gesture: AICameraFineHandGesture,
    effective_visible_person_boxes: tuple[AICameraBox, ...],
    effective_hand_boxes: tuple[_HandBox, ...],
    person_roi_detection_count: int,
    person_roi_combined_gesture: str,
    live_roi_hand_box_count: int,
    live_roi_combined_gesture: str,
) -> str | None:
    if fine_hand_gesture != AICameraFineHandGesture.NONE:
        return None
    if not effective_visible_person_boxes:
        if not effective_hand_boxes:
            return "no_person_roi_or_live_hand_box"
        if (
            live_roi_hand_box_count > 0
            and live_roi_combined_gesture == AICameraFineHandGesture.NONE.value
        ):
            return "live_hand_roi_without_symbol"
        return None
    if (
        live_roi_hand_box_count > 0
        and live_roi_combined_gesture == AICameraFineHandGesture.NONE.value
    ):
        return "live_hand_roi_without_symbol"
    if person_roi_detection_count <= 0:
        return "visible_person_roi_without_hand_detection"
    if person_roi_combined_gesture == AICameraFineHandGesture.NONE.value:
        return "visible_person_roi_with_hand_detection_without_symbol"
    return None


def _full_frame_rescue_allowed_for_reason(
    *,
    observe_policy: LiveGestureObservePolicy,
    reason: str,
) -> bool:
    if observe_policy.allow_full_frame_hand_recovery:
        return True
    return (
        reason == "visible_person_roi_with_hand_detection_without_symbol"
        and observe_policy.allow_full_frame_hand_recovery_after_person_roi_detection
    )


def _coerce_roi_source_value(value: object) -> str:
    raw = getattr(value, "value", None) or str(value or "")
    return str(raw).strip()


def _person_roi_source_priority(roi_source: str) -> int:
    return int(_PERSON_ROI_SOURCE_PRIORITY.get(roi_source, 0))


def _person_roi_source_min_confidence(roi_source: str) -> float | None:
    value = _PERSON_ROI_SOURCE_MIN_CONFIDENCE.get(roi_source)
    return _coerce_unit_interval(value) if value is not None else None


def _effective_person_roi_source_min_confidence(
    *,
    roi_source: str,
    gesture_frame_source: str,
) -> float | None:
    if gesture_frame_source == "full_frame_landmark_crop":
        return None
    return _person_roi_source_min_confidence(roi_source)


def _person_roi_short_circuit_min_confidence(config: MediaPipeVisionConfig) -> float:
    builtin_min_score = _coerce_unit_interval(getattr(config, "builtin_gesture_min_score", None)) or 0.0
    custom_min_score = _live_custom_min_score(config) if _live_custom_gesture_enabled(config) else 0.0
    return max(builtin_min_score, custom_min_score, _LIVE_CUSTOM_MIN_SCORE_FLOOR)


def _should_short_circuit_person_roi_scan(
    *,
    config: MediaPipeVisionConfig,
    pose_hint_attached: bool,
    candidate: _PersonRoiGestureChoice,
) -> bool:
    if not pose_hint_attached or candidate.gesture == AICameraFineHandGesture.NONE:
        return False
    confidence = _coerce_unit_interval(candidate.confidence)
    if confidence is None:
        return False
    return confidence >= _person_roi_short_circuit_min_confidence(config)


def _guard_person_roi_gesture_choice(
    *,
    candidate: _GestureChoice,
    roi_source: str,
    gesture_frame_source: str,
    detection_index: int,
) -> _PersonRoiGestureChoice:
    gesture, confidence = candidate
    score = _coerce_unit_interval(confidence)
    required_score = _effective_person_roi_source_min_confidence(
        roi_source=roi_source,
        gesture_frame_source=gesture_frame_source,
    )
    if gesture == AICameraFineHandGesture.NONE or score is None:
        return _PersonRoiGestureChoice(
            roi_source=roi_source,
            detection_index=detection_index,
        )
    if required_score is not None and score < required_score:
        return _PersonRoiGestureChoice(
            roi_source=roi_source,
            detection_index=detection_index,
        )
    return _PersonRoiGestureChoice(
        gesture=gesture,
        confidence=confidence,
        roi_source=roi_source,
        detection_index=detection_index,
    )


def _prefer_person_roi_gesture_choice(
    current: _PersonRoiGestureChoice,
    challenger: _PersonRoiGestureChoice,
) -> _PersonRoiGestureChoice:
    if challenger.gesture == AICameraFineHandGesture.NONE:
        return current
    if current.gesture == AICameraFineHandGesture.NONE:
        return challenger
    current_score = _coerce_unit_interval(current.confidence) or 0.0
    challenger_score = _coerce_unit_interval(challenger.confidence) or 0.0
    current_priority = _person_roi_source_priority(current.roi_source)
    challenger_priority = _person_roi_source_priority(challenger.roi_source)

    if challenger_priority > current_priority and challenger_score >= (current_score - _PERSON_ROI_SOURCE_PRIORITY_MARGIN):
        return challenger
    if current_priority > challenger_priority and current_score >= (challenger_score - _PERSON_ROI_SOURCE_PRIORITY_MARGIN):
        return current
    if challenger_score > current_score:
        return challenger
    if current_score > challenger_score:
        return current
    if challenger_priority > current_priority:
        return challenger
    return current


def _combine_person_roi_gesture_choices(
    *,
    builtin: _PersonRoiGestureChoice,
    custom: _PersonRoiGestureChoice,
) -> _PersonRoiGestureChoice:
    combined = combine_task_specific_custom_gesture_choice(
        builtin.as_choice(),
        custom.as_choice(),
        preferred_custom_gestures=_LIVE_FINE_GESTURE_ALLOWLIST,
    )
    if combined[0] == AICameraFineHandGesture.NONE:
        return _PersonRoiGestureChoice()
    if combined == custom.as_choice() and custom.gesture != AICameraFineHandGesture.NONE:
        return custom
    return builtin


def _open_palm_score(categories: object) -> float | None:
    try:
        candidates = list(cast(Any, categories))
    except TypeError:
        return None

    best_score = None
    for category in candidates:
        label = str(getattr(category, "category_name", "") or "").strip().lower()
        if label != "open_palm":
            continue
        try:
            score = float(cast(Any, getattr(category, "score", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        score = max(0.0, min(1.0, score))
        if best_score is None or score > best_score:
            best_score = score
    return best_score


def _hand_center_x(landmarks: object) -> float | None:
    try:
        points = list(cast(Any, landmarks))
    except TypeError:
        return None
    xs: list[float] = []
    for point in points:
        try:
            value = float(cast(Any, getattr(point, "x", None)))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            xs.append(max(0.0, min(1.0, value)))
    if not xs:
        return None
    return sum(xs) / len(xs)


def _coerce_unit_interval(value: object) -> float | None:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def _round_optional_confidence(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(max(0.0, min(1.0, float(value))), 3)


def _normalize_hand_box(box: _HandBox) -> _HandBox | None:
    try:
        top = float(box[0])
        left = float(box[1])
        bottom = float(box[2])
        right = float(box[3])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in (top, left, bottom, right)):
        return None
    top = max(0.0, min(1.0, top))
    left = max(0.0, min(1.0, left))
    bottom = max(0.0, min(1.0, bottom))
    right = max(0.0, min(1.0, right))
    if bottom <= top or right <= left:
        return None
    return (top, left, bottom, right)


def _hand_box_iou(a: _HandBox, b: _HandBox) -> float:
    inter_top = max(a[0], b[0])
    inter_left = max(a[1], b[1])
    inter_bottom = min(a[2], b[2])
    inter_right = min(a[3], b[3])
    inter_h = max(0.0, inter_bottom - inter_top)
    inter_w = max(0.0, inter_right - inter_left)
    inter_area = inter_h * inter_w
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _merge_hand_boxes(
    builtin_boxes: tuple[_HandBox, ...],
    custom_boxes: tuple[_HandBox, ...],
) -> tuple[_HandBox, ...]:
    merged: list[_HandBox] = []
    for raw_box in tuple(builtin_boxes or ()) + tuple(custom_boxes or ()):
        normalized = _normalize_hand_box(raw_box)
        if normalized is None:
            continue
        duplicate = False
        for existing in merged:
            if _hand_box_iou(existing, normalized) >= _HAND_BOX_DUPLICATE_IOU:
                duplicate = True
                break
        if not duplicate:
            merged.append(normalized)
    return tuple(merged)


def _box_area(box: Any) -> float:
    try:
        top = float(getattr(box, "top", box[0]))
        left = float(getattr(box, "left", box[1]))
        bottom = float(getattr(box, "bottom", box[2]))
        right = float(getattr(box, "right", box[3]))
    except Exception:
        return 0.0
    return max(0.0, bottom - top) * max(0.0, right - left)


def _hand_box_area(box: _HandBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _hand_box_center_distance(box: _HandBox) -> float:
    center_y = (box[0] + box[2]) / 2.0
    center_x = (box[1] + box[3]) / 2.0
    return math.hypot(center_x - 0.5, center_y - 0.5)


def _prioritize_person_boxes(
    *,
    primary_person_box: AICameraBox | None,
    person_boxes: tuple[AICameraBox, ...],
    limit: int,
) -> tuple[AICameraBox, ...]:
    if limit <= 0 or not person_boxes:
        return ()
    ranked: list[tuple[int, float, float, int, AICameraBox]] = []
    for index, box in enumerate(person_boxes):
        exact = 1 if (primary_person_box is not None and box == primary_person_box) else 0
        overlap = 0.0
        if primary_person_box is not None:
            try:
                overlap = max(0.0, min(1.0, float(iou(box, primary_person_box))))
            except Exception:
                overlap = 0.0
        ranked.append((exact, overlap, _box_area(box), -index, box))
    ranked.sort(reverse=True)
    return tuple(item[-1] for item in ranked[:limit])


def _prioritize_hand_boxes(
    *,
    hand_boxes: tuple[_HandBox, ...],
    limit: int,
) -> tuple[_HandBox, ...]:
    if limit <= 0 or not hand_boxes:
        return ()
    ranked = sorted(
        hand_boxes,
        key=lambda box: (-_hand_box_area(box), _hand_box_center_distance(box)),
    )
    return tuple(ranked[:limit])


def _coerce_nonnegative_int(value: object, *, default: int = 0) -> int:
    try:
        numeric = int(cast(Any, value))
    except (TypeError, ValueError):
        return default
    return max(0, numeric)


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    try:
        normalized = str(value).strip().lower()
    except Exception:
        return default
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_bounded_float(
    value: object,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(minimum, min(maximum, numeric))


def _coerce_bounded_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        numeric = int(cast(Any, value))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))


def _live_result_history_size(config: MediaPipeVisionConfig) -> int:
    return _coerce_bounded_int(
        getattr(config, "live_result_history_size", None),
        default=_DEFAULT_RESULT_HISTORY_SIZE,
        minimum=4,
        maximum=64,
    )


def _live_max_person_roi_candidates(config: MediaPipeVisionConfig) -> int:
    return _coerce_bounded_int(
        getattr(config, "live_max_person_roi_candidates", None),
        default=_DEFAULT_MAX_PERSON_ROI_CANDIDATES,
        minimum=1,
        maximum=8,
    )


def _live_max_hand_roi_candidates(config: MediaPipeVisionConfig) -> int:
    return _coerce_bounded_int(
        getattr(config, "live_max_hand_roi_candidates", None),
        default=_DEFAULT_MAX_HAND_ROI_CANDIDATES,
        minimum=1,
        maximum=6,
    )


def _live_full_frame_rescue_min_interval_s(config: MediaPipeVisionConfig) -> float:
    return _coerce_bounded_float(
        getattr(config, "live_full_frame_rescue_min_interval_s", None),
        default=_DEFAULT_FULL_FRAME_RESCUE_MIN_INTERVAL_S,
        minimum=0.0,
        maximum=2.0,
    )


def _live_pending_result_timeout_s(config: MediaPipeVisionConfig) -> float:
    return _coerce_bounded_float(
        getattr(config, "live_pending_result_timeout_s", None),
        default=_DEFAULT_MAX_PENDING_LIVE_RESULT_AGE_S,
        minimum=max(_DEFAULT_CURRENT_RESULT_WAIT_S, 0.25),
        maximum=5.0,
    )


def _live_temporal_stability_enabled(config: MediaPipeVisionConfig) -> bool:
    return _coerce_bool(
        getattr(config, "live_temporal_stability_enabled", None),
        default=False,
    )


def _live_temporal_window_s(config: MediaPipeVisionConfig) -> float:
    return _coerce_bounded_float(
        getattr(config, "live_temporal_window_s", None),
        default=_TEMPORAL_WINDOW_S,
        minimum=0.10,
        maximum=2.0,
    )


def _live_temporal_window_samples(config: MediaPipeVisionConfig) -> int:
    return _coerce_bounded_int(
        getattr(config, "live_temporal_window_samples", None),
        default=_TEMPORAL_WINDOW_SAMPLES,
        minimum=2,
        maximum=8,
    )


def _live_temporal_required_votes(config: MediaPipeVisionConfig) -> int:
    requested = _coerce_bounded_int(
        getattr(config, "live_temporal_required_votes", None),
        default=_TEMPORAL_REQUIRED_VOTES,
        minimum=1,
        maximum=8,
    )
    return min(requested, _live_temporal_window_samples(config))


def _live_temporal_strong_confidence(config: MediaPipeVisionConfig) -> float:
    return _coerce_bounded_float(
        getattr(config, "live_temporal_strong_confidence", None),
        default=_TEMPORAL_STRONG_CONFIDENCE,
        minimum=0.50,
        maximum=0.99,
    )


def _live_temporal_hold_s(config: MediaPipeVisionConfig) -> float:
    return _coerce_bounded_float(
        getattr(config, "live_temporal_hold_s", None),
        default=_TEMPORAL_HOLD_S,
        minimum=0.0,
        maximum=1.0,
    )


def _short_exception(exc: Exception, *, limit: int = 160) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    return text[:limit]


__all__ = [
    "LiveGestureFrameObservation",
    "LiveGestureObservePolicy",
    "LiveGesturePipeline",
]
