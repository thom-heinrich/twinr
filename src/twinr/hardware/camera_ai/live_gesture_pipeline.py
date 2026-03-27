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

from collections import deque
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
from .mediapipe_runtime import MediaPipeTaskRuntime
from .models import AICameraBox, AICameraFineHandGesture, AICameraGestureEvent


_DEFAULT_MAX_RESULT_AGE_S = 0.75
_DEFAULT_CURRENT_RESULT_WAIT_S = 0.2
_LIVE_CUSTOM_MIN_SCORE_FLOOR = 0.6
_WAVE_WINDOW_S = 0.55
_WAVE_MIN_SAMPLES = 3
_WAVE_MIN_SPAN_X = 0.10
_WAVE_MIN_TRAVEL_X = 0.18
_WAVE_DIRECTION_DELTA = 0.015
_RECENT_PRIMARY_PERSON_BOX_TTL_S = 0.45
_RECENT_VISIBLE_PERSON_BOXES_TTL_S = 0.45
_RECENT_HAND_BOXES_TTL_S = 0.45
_LIVE_GESTURE_NUM_HANDS = 1
_PRIMARY_PERSON_HINT_MIN_IOU = 0.6
_LIVE_HAND_CROP_PADDING = 0.14
_LIVE_HAND_MIN_CONTEXT_RATIO = 0.18
_PERSON_ROI_SOURCE_PRIORITY_MARGIN = 0.12
_PERSON_ROI_POSE_HINT_SHORT_CIRCUIT_REASON = "primary_pose_hint_confident_gesture"
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
_LIVE_NO_HAND_PERSON_ROI_ALLOWLIST = frozenset(
    {
        AICameraFineHandGesture.PEACE_SIGN,
    }
)
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


@dataclass(frozen=True, slots=True)
class _RecognizerSnapshot:
    """Store the newest live-stream recognizer callback payload."""

    timestamp_ms: int
    gesture: AICameraFineHandGesture
    confidence: float | None
    hand_count: int
    open_palm_center_x: float | None
    hand_boxes: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class _WaveSample:
    """Store one open-palm motion sample for wave tracking."""

    observed_at: float
    center_x: float


@dataclass(frozen=True, slots=True)
class _PersonRoiGestureChoice:
    """Store one person-ROI gesture candidate with its source provenance."""

    gesture: AICameraFineHandGesture = AICameraFineHandGesture.NONE
    confidence: float | None = None
    roi_source: str = ""
    detection_index: int | None = None

    def as_choice(self) -> tuple[AICameraFineHandGesture, float | None]:
        """Expose the public gesture tuple used by the rest of the pipeline."""

        return self.gesture, self.confidence


class _WaveGestureTracker:
    """Detect a short open-palm lateral wave over recent callback samples."""

    def __init__(self) -> None:
        self._samples: deque[_WaveSample] = deque()

    def observe(
        self,
        *,
        observed_at: float,
        open_palm_center_x: float | None,
    ) -> tuple[AICameraGestureEvent, float | None]:
        """Update the tracker and return one optional bounded wave event."""

        self._prune(observed_at)
        if open_palm_center_x is None:
            return AICameraGestureEvent.NONE, None
        self._samples.append(
            _WaveSample(
                observed_at=observed_at,
                center_x=open_palm_center_x,
            )
        )
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
        confidence = min(
            0.99,
            0.55 + min(0.25, span_x * 1.2) + min(0.19, total_travel_x * 0.6),
        )
        return AICameraGestureEvent.WAVE, round(confidence, 3)

    def _prune(self, observed_at: float) -> None:
        cutoff = observed_at - _WAVE_WINDOW_S
        while self._samples and self._samples[0].observed_at < cutoff:
            self._samples.popleft()


class LiveGesturePipeline:
    """Submit RGB frames to MediaPipe LIVE_STREAM gesture recognizers."""

    def __init__(
        self,
        *,
        config: MediaPipeVisionConfig,
    ) -> None:
        self.config = config
        self._runtime = MediaPipeTaskRuntime(config=config)
        self._lock = RLock()
        self._result_condition = Condition(self._lock)
        self._latest_builtin: _RecognizerSnapshot | None = None
        self._latest_custom: _RecognizerSnapshot | None = None
        self._wave_tracker = _WaveGestureTracker()
        self._hand_landmark_worker: MediaPipeHandLandmarkWorker | None = None
        self._recent_primary_person_box: AICameraBox | None = None
        self._recent_primary_person_seen_at: float | None = None
        self._recent_visible_person_boxes: tuple[AICameraBox, ...] = ()
        self._recent_visible_person_boxes_seen_at: float | None = None
        self._recent_hand_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self._recent_hand_boxes_seen_at: float | None = None
        self._last_debug_snapshot: dict[str, object] = {}

    def close(self) -> None:
        """Close live-stream recognizers and reset cached callback state."""

        with self._lock:
            self._latest_builtin = None
            self._latest_custom = None
            self._wave_tracker = _WaveGestureTracker()
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
        """Return the newest bounded debug snapshot for this live gesture lane."""

        with self._lock:
            return dict(self._last_debug_snapshot)

    def observe(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
        primary_person_box: AICameraBox | None = None,
        visible_person_boxes: tuple[AICameraBox, ...] = (),
        person_count: int = 0,
        sparse_keypoints: dict[int, tuple[float, float, float]] | None = None,
    ) -> LiveGestureFrameObservation:
        """Submit one RGB frame and return the freshest bounded live result."""

        with workflow_span(
            name="live_gesture_pipeline_observe",
            kind="io",
            details={
                "person_count": max(0, int(person_count)),
                "visible_person_box_count": len(tuple(visible_person_boxes or ())),
                "pose_hint_keypoint_count": len(dict(sparse_keypoints or {})),
            },
        ):
            runtime = self._runtime.load_runtime()
            image = self._runtime.build_image(runtime, frame_rgb=frame_rgb)
            timestamp_ms = self._runtime.timestamp_ms(observed_at)
            live_custom_enabled = _live_custom_gesture_enabled(self.config)
            builtin_recognizer = self._runtime.ensure_live_gesture_recognizer(
                runtime,
                result_callback=self._handle_builtin_result,
                num_hands_override=_LIVE_GESTURE_NUM_HANDS,
            )
            builtin_recognizer.recognize_async(image, timestamp_ms)
            if live_custom_enabled:
                custom_recognizer = self._runtime.ensure_live_custom_gesture_recognizer(
                    runtime,
                    result_callback=self._handle_custom_result,
                )
                custom_recognizer.recognize_async(image, timestamp_ms)
            (
                live_result_wait_s,
                current_live_builtin_ready,
                current_live_custom_ready,
            ) = self._await_current_live_results(
                timestamp_ms=timestamp_ms,
                expect_custom=live_custom_enabled,
            )
            return self._build_observation(
                runtime=runtime,
                frame_rgb=frame_rgb,
                observed_at=observed_at,
                timestamp_ms=timestamp_ms,
                live_result_wait_s=live_result_wait_s,
                current_live_builtin_ready=current_live_builtin_ready,
                current_live_custom_ready=(
                    current_live_custom_ready if live_custom_enabled else None
                ),
                primary_person_box=primary_person_box,
                visible_person_boxes=visible_person_boxes,
                person_count=person_count,
                sparse_keypoints=dict(sparse_keypoints or {}),
            )

    def _build_observation(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        observed_at: float,
        timestamp_ms: int,
        live_result_wait_s: float,
        current_live_builtin_ready: bool,
        current_live_custom_ready: bool | None,
        primary_person_box: AICameraBox | None,
        visible_person_boxes: tuple[AICameraBox, ...],
        person_count: int,
        sparse_keypoints: dict[int, tuple[float, float, float]],
    ) -> LiveGestureFrameObservation:
        """Materialize one bounded observation from the newest callback results."""

        with self._lock:
            builtin = self._fresh_snapshot(self._latest_builtin, observed_at=observed_at)
            custom = self._fresh_snapshot(self._latest_custom, observed_at=observed_at)
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
            open_palm_center_x = None
            if builtin is not None and builtin.gesture == AICameraFineHandGesture.OPEN_PALM:
                open_palm_center_x = builtin.open_palm_center_x
            gesture_event, gesture_confidence = self._wave_tracker.observe(
                observed_at=observed_at,
                open_palm_center_x=open_palm_center_x,
            )
            freshest_timestamp_ms = max(
                0 if builtin is None else builtin.timestamp_ms,
                0 if custom is None else custom.timestamp_ms,
            )
            current_hand_boxes = _merge_hand_boxes(
                () if builtin is None else builtin.hand_boxes,
                () if custom is None else custom.hand_boxes,
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
            if effective_primary_person_box is None:
                effective_primary_person_box = self._fresh_recent_primary_person_box(observed_at)
                if effective_primary_person_box is not None:
                    primary_person_box_source = "recent"
            effective_visible_person_boxes = live_visible_person_boxes
            visible_person_box_source = "live" if live_visible_person_boxes and person_count > 0 else "none"
            if not effective_visible_person_boxes:
                effective_visible_person_boxes = self._fresh_recent_visible_person_boxes(observed_at)
                if effective_visible_person_boxes:
                    visible_person_box_source = "recent"
            if not effective_visible_person_boxes and effective_primary_person_box is not None:
                effective_visible_person_boxes = (effective_primary_person_box,)
                visible_person_box_source = primary_person_box_source
            if current_hand_boxes:
                self._recent_hand_boxes = tuple(current_hand_boxes)
                self._recent_hand_boxes_seen_at = observed_at
                effective_hand_boxes = tuple(current_hand_boxes)
                hand_box_source = "live"
            else:
                effective_hand_boxes = self._fresh_recent_hand_boxes(observed_at)
                hand_box_source = "recent" if effective_hand_boxes else "none"
        result_age_s = None
        if freshest_timestamp_ms > 0:
            result_age_s = max(0.0, observed_at - (freshest_timestamp_ms / 1000.0))
        fresh_live_results_confirm_no_hand = _fresh_live_results_confirm_no_hand(
            builtin_ready=current_live_builtin_ready,
            custom_ready=current_live_custom_ready,
            hand_count=hand_count,
            live_hand_box_count=len(current_hand_boxes),
        )
        if fine_hand_gesture == AICameraFineHandGesture.NONE:
            rescue_blocked_by_live_no_hand = fresh_live_results_confirm_no_hand and not effective_hand_boxes
            debug_snapshot: dict[str, object] = {
                "resolved_source": "none",
                "live_builtin_gesture": builtin_choice[0].value,
                "live_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_custom_gesture": custom_choice[0].value,
                "live_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "live_custom_enabled": _live_custom_gesture_enabled(self.config),
                "live_hand_count": hand_count,
                "live_hand_box_count": len(current_hand_boxes),
                "effective_live_hand_box_count": len(effective_hand_boxes),
                "live_hand_box_source": hand_box_source,
                "live_result_age_s": _round_optional_confidence(result_age_s),
                "current_live_result_wait_s": _round_optional_confidence(live_result_wait_s),
                "current_live_builtin_ready": current_live_builtin_ready,
                "current_live_custom_ready": current_live_custom_ready,
                "fresh_live_results_confirm_no_hand": fresh_live_results_confirm_no_hand,
                "input_person_count": max(0, int(person_count)),
                "primary_person_box_available": primary_person_box is not None,
                "effective_primary_person_box_available": effective_primary_person_box is not None,
                "primary_person_box_source": primary_person_box_source,
                "pose_hint_keypoint_count": len(sparse_keypoints),
                "visible_person_box_count": len(live_visible_person_boxes),
                "effective_visible_person_box_count": len(effective_visible_person_boxes),
                "visible_person_box_source": visible_person_box_source,
                "person_roi_candidate_count": len(effective_visible_person_boxes),
                "person_roi_detection_count": 0,
                "person_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
                "person_roi_combined_confidence": None,
                "person_roi_short_circuit_used": False,
                "person_roi_short_circuit_reason": None,
                "person_roi_block_reason": None,
                "full_frame_hand_attempt_reason": None,
            }
            person_roi_choice: _GestureChoice = (AICameraFineHandGesture.NONE, None)
            if effective_visible_person_boxes:
                person_roi_choice, person_roi_debug = self._recognize_from_person_rois(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    primary_person_box=effective_primary_person_box,
                    person_boxes=effective_visible_person_boxes,
                    sparse_keypoints=sparse_keypoints,
                )
                debug_snapshot.update(person_roi_debug)
                if rescue_blocked_by_live_no_hand and not _allow_person_roi_rescue_despite_live_no_hand(
                    person_roi_choice[0]
                ):
                    debug_snapshot["person_roi_block_reason"] = "fresh_live_results_confirm_no_hand"
                elif person_roi_choice[0] != AICameraFineHandGesture.NONE:
                    fine_hand_gesture, fine_hand_confidence = person_roi_choice
                    if len(effective_visible_person_boxes) > 1 and visible_person_box_source in {"live", "recent"}:
                        debug_snapshot["resolved_source"] = (
                            "visible_person_roi"
                            if visible_person_box_source == "live"
                            else "recent_visible_person_roi"
                        )
                    else:
                        debug_snapshot["resolved_source"] = (
                            "person_roi" if visible_person_box_source == "live" else "recent_person_roi"
                        )
            elif rescue_blocked_by_live_no_hand:
                debug_snapshot["person_roi_block_reason"] = "fresh_live_results_confirm_no_hand"
            if fine_hand_gesture == AICameraFineHandGesture.NONE:
                rescue_choice, live_roi_debug = self._recognize_from_live_hand_rois(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    hand_boxes=effective_hand_boxes,
                    hand_box_source=hand_box_source,
                )
                debug_snapshot.update(live_roi_debug)
                if rescue_choice[0] != AICameraFineHandGesture.NONE:
                    fine_hand_gesture, fine_hand_confidence = rescue_choice
                    debug_snapshot["resolved_source"] = (
                        "live_hand_roi" if hand_box_source == "live" else "recent_live_hand_roi"
                    )
            if rescue_blocked_by_live_no_hand and fine_hand_gesture == AICameraFineHandGesture.NONE:
                debug_snapshot["full_frame_hand_attempt_reason"] = "fresh_live_results_confirm_no_hand"
            else:
                full_frame_rescue_reason = _resolve_full_frame_rescue_reason(
                    fine_hand_gesture=fine_hand_gesture,
                    effective_visible_person_boxes=effective_visible_person_boxes,
                    effective_hand_boxes=effective_hand_boxes,
                    person_roi_detection_count=_coerce_nonnegative_int(
                        debug_snapshot.get("person_roi_detection_count")
                    ),
                    live_roi_hand_box_count=_coerce_nonnegative_int(
                        debug_snapshot.get("live_roi_hand_box_count")
                    ),
                    live_roi_combined_gesture=str(
                        debug_snapshot.get("live_roi_combined_gesture", AICameraFineHandGesture.NONE.value)
                        or AICameraFineHandGesture.NONE.value
                    ),
                )
                if full_frame_rescue_reason is not None:
                    full_frame_choice, full_frame_debug = self._recognize_from_full_frame_hand_landmarks(
                        runtime=runtime,
                        frame_rgb=frame_rgb,
                        timestamp_ms=timestamp_ms,
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
        else:
            debug_snapshot = {
                "resolved_source": "live_stream",
                "live_builtin_gesture": builtin_choice[0].value,
                "live_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_custom_gesture": custom_choice[0].value,
                "live_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "live_custom_enabled": _live_custom_gesture_enabled(self.config),
                "live_hand_count": hand_count,
                "live_hand_box_count": len(current_hand_boxes),
                "live_result_age_s": _round_optional_confidence(result_age_s),
                "current_live_result_wait_s": _round_optional_confidence(live_result_wait_s),
                "current_live_builtin_ready": current_live_builtin_ready,
                "current_live_custom_ready": current_live_custom_ready,
                "fresh_live_results_confirm_no_hand": fresh_live_results_confirm_no_hand,
                "input_person_count": max(0, int(person_count)),
                "primary_person_box_available": primary_person_box is not None,
                "effective_primary_person_box_available": effective_primary_person_box is not None,
                "primary_person_box_source": primary_person_box_source,
                "pose_hint_keypoint_count": len(sparse_keypoints),
                "visible_person_box_count": len(live_visible_person_boxes),
                "effective_visible_person_box_count": len(effective_visible_person_boxes),
                "visible_person_box_source": visible_person_box_source,
            }
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
    ) -> tuple[tuple[AICameraFineHandGesture, float | None], dict[str, object]]:
        """Recover concrete symbols from bounded visible-person upper-body ROIs."""

        with workflow_span(
            name="live_gesture_pipeline_person_roi_recovery",
            kind="decision",
            details={"person_roi_candidate_count": len(person_boxes)},
        ):
            debug = {
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
                hand_landmark_result = self._ensure_hand_landmark_worker().analyze(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms + index,
                    primary_person_box=person_box,
                    sparse_keypoints=candidate_sparse_keypoints,
                )
                detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
                debug["person_roi_detection_count"] = (
                    _coerce_nonnegative_int(debug.get("person_roi_detection_count")) + len(detections)
                )
                (
                    builtin_choice,
                    custom_choice,
                    detection_debug,
                ) = self._recognize_from_hand_landmark_result(
                    runtime=runtime,
                    hand_landmark_result=hand_landmark_result,
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
                            "builtin_source": builtin_choice.roi_source or None,
                            "custom_gesture": custom_choice.gesture.value,
                            "custom_confidence": _round_optional_confidence(custom_choice.confidence),
                            "custom_source": custom_choice.roi_source or None,
                            "combined_gesture": combined.gesture.value,
                            "combined_confidence": _round_optional_confidence(combined.confidence),
                            "combined_source": combined.roi_source or None,
                            "pose_hint_attached": pose_hint_attached,
                        },
                        "constraints_violated": (
                            []
                            if combined.gesture != AICameraFineHandGesture.NONE
                            else ["no_symbol_or_source_guard"]
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
                    debug["person_roi_short_circuit_confidence"] = _round_optional_confidence(
                        combined.confidence
                    )
                    debug["person_roi_short_circuit_reason"] = (
                        _PERSON_ROI_POSE_HINT_SHORT_CIRCUIT_REASON
                    )
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
        """Attach pose hints only to the ROI that still matches the tracked primary person."""

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
    ) -> tuple[
        _PersonRoiGestureChoice,
        _PersonRoiGestureChoice,
        tuple[dict[str, object], ...],
    ]:
        """Run bounded image-mode gesture recognition on hand-ROI landmark crops."""

        detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
        if not detections:
            return (
                _PersonRoiGestureChoice(),
                _PersonRoiGestureChoice(),
                (),
            )

        builtin_choice = _PersonRoiGestureChoice()
        custom_choice = _PersonRoiGestureChoice()
        detection_debug: list[dict[str, object]] = []
        builtin_recognizer = self._runtime.ensure_roi_gesture_recognizer(runtime)
        custom_recognizer = None
        if _live_custom_gesture_enabled(self.config):
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
            next_builtin_choice = _prefer_person_roi_gesture_choice(
                builtin_choice,
                builtin_candidate,
            )
            builtin_choice = next_builtin_choice
            custom_result = None
            custom_raw_candidate: _GestureChoice = (AICameraFineHandGesture.NONE, None)
            custom_candidate = _PersonRoiGestureChoice(
                roi_source=roi_source,
                detection_index=index,
            )
            if custom_recognizer is None:
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
                        custom_context_retry_used=False,
                        custom_context_candidate=(AICameraFineHandGesture.NONE, None),
                        custom_context_result=None,
                    )
                )
                continue
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
            next_custom_choice = _prefer_person_roi_gesture_choice(
                custom_choice,
                custom_candidate,
            )
            custom_choice = next_custom_choice
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
        return (
            builtin_choice,
            custom_choice,
            tuple(detection_debug),
        )

    def _handle_builtin_result(
        self,
        result: object,
        _output_image: object,
        timestamp_ms: int,
    ) -> None:
        """Store one built-in live-stream recognizer callback result."""

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
            if self._latest_builtin is None or snapshot.timestamp_ms >= self._latest_builtin.timestamp_ms:
                self._latest_builtin = snapshot
            self._result_condition.notify_all()

    def _handle_custom_result(
        self,
        result: object,
        _output_image: object,
        timestamp_ms: int,
    ) -> None:
        """Store one custom live-stream recognizer callback result."""

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
            if self._latest_custom is None or snapshot.timestamp_ms >= self._latest_custom.timestamp_ms:
                self._latest_custom = snapshot
            self._result_condition.notify_all()

    def _recognize_from_live_hand_rois(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        hand_boxes: tuple[tuple[float, float, float, float], ...],
        hand_box_source: str,
    ) -> tuple[tuple[AICameraFineHandGesture, float | None], dict[str, object]]:
        """Recover concrete symbols from tight hand crops when live labels stay empty.

        The full-frame live recognizer is the fastest path, but on the Pi it can
        already localize a hand while still labeling the frame as `none`.
        Reusing those live hand landmarks for tight IMAGE-mode ROI gesture
        recognition recovers explicit symbols without pulling the hot path back
        through the heavier social-vision pipeline.
        """

        debug = {
            "live_roi_hand_box_count": 0,
            "live_roi_hand_box_source": hand_box_source,
            "live_roi_detection_debug": (),
            "live_roi_builtin_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_builtin_confidence": None,
            "live_roi_custom_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_custom_confidence": None,
            "live_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
            "live_roi_combined_confidence": None,
        }
        debug["live_roi_hand_box_count"] = len(hand_boxes)
        if not hand_boxes:
            return (AICameraFineHandGesture.NONE, None), debug

        builtin_recognizer = self._runtime.ensure_roi_gesture_recognizer(runtime)
        custom_recognizer = None
        if _live_custom_gesture_enabled(self.config):
            custom_recognizer = self._runtime.ensure_custom_roi_gesture_recognizer(runtime)

        builtin_choice: _GestureChoice = (AICameraFineHandGesture.NONE, None)
        custom_choice: _GestureChoice = (AICameraFineHandGesture.NONE, None)
        hand_box_options: list[dict[str, object]] = []
        hand_box_debug: list[dict[str, object]] = []
        for index, hand_box in enumerate(hand_boxes):
            crop = _crop_hand_box(frame_rgb, hand_box)
            if crop is None:
                continue
            image = self._runtime.build_image(runtime, frame_rgb=crop)
            builtin_result = builtin_recognizer.recognize(image)
            builtin_candidate = resolve_fine_hand_gesture(
                result=builtin_result,
                category_map=_LIVE_BUILTIN_FINE_GESTURE_MAP,
                min_score=self.config.builtin_gesture_min_score,
            )
            builtin_choice = prefer_gesture_choice(
                builtin_choice,
                builtin_candidate,
            )
            custom_result = None
            custom_candidate: _GestureChoice = (AICameraFineHandGesture.NONE, None)
            if custom_recognizer is None:
                hand_box_debug.append(
                    {
                        "hand_box_index": index,
                        "builtin_gesture": builtin_candidate[0].value,
                        "builtin_confidence": _round_optional_confidence(builtin_candidate[1]),
                        "builtin_categories": _summarize_gesture_categories(builtin_result),
                    }
                )
                hand_box_options.append(
                    {
                        "id": f"live_hand_roi_{index}",
                        "summary": f"Hand ROI {index} produced {builtin_candidate[0].value}.",
                        "score_components": {
                            "builtin_gesture": builtin_candidate[0].value,
                            "builtin_confidence": _round_optional_confidence(builtin_candidate[1]),
                        },
                        "constraints_violated": (
                            [] if builtin_candidate[0] != AICameraFineHandGesture.NONE else ["no_symbol"]
                        ),
                    }
                )
                continue
            custom_result = custom_recognizer.recognize(image)
            custom_candidate = resolve_fine_hand_gesture(
                result=custom_result,
                category_map=_LIVE_CUSTOM_FINE_GESTURE_MAP,
                min_score=_live_custom_min_score(self.config),
            )
            custom_choice = prefer_gesture_choice(
                custom_choice,
                custom_candidate,
            )
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
    ) -> tuple[tuple[AICameraFineHandGesture, float | None], dict[str, object]]:
        """Recover one symbol from a final bounded whole-frame hand pass."""

        debug = {
            "full_frame_hand_detection_count": 0,
            "full_frame_hand_detection_debug": (),
            "full_frame_hand_builtin_gesture": AICameraFineHandGesture.NONE.value,
            "full_frame_hand_builtin_confidence": None,
            "full_frame_hand_custom_gesture": AICameraFineHandGesture.NONE.value,
            "full_frame_hand_custom_confidence": None,
            "full_frame_hand_combined_gesture": AICameraFineHandGesture.NONE.value,
            "full_frame_hand_combined_confidence": None,
        }
        hand_landmark_result = self._ensure_hand_landmark_worker().analyze_full_frame(
            runtime=runtime,
            frame_rgb=frame_rgb,
            timestamp_ms=timestamp_ms,
        )
        detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
        debug["full_frame_hand_detection_count"] = len(detections)
        builtin_choice, custom_choice, detection_debug = self._recognize_from_hand_landmark_result(
            runtime=runtime,
            hand_landmark_result=hand_landmark_result,
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

    def _fresh_recent_primary_person_box(
        self,
        observed_at: float,
    ) -> AICameraBox | None:
        """Reuse the newest primary-person ROI hint for short detection flickers only."""

        if self._recent_primary_person_box is None or self._recent_primary_person_seen_at is None:
            return None
        if (observed_at - self._recent_primary_person_seen_at) > _RECENT_PRIMARY_PERSON_BOX_TTL_S:
            return None
        return self._recent_primary_person_box

    def _fresh_recent_visible_person_boxes(
        self,
        observed_at: float,
    ) -> tuple[AICameraBox, ...]:
        """Reuse newest visible-person ROI hints for short detection flickers only."""

        if not self._recent_visible_person_boxes or self._recent_visible_person_boxes_seen_at is None:
            return ()
        if (observed_at - self._recent_visible_person_boxes_seen_at) > _RECENT_VISIBLE_PERSON_BOXES_TTL_S:
            return ()
        return tuple(self._recent_visible_person_boxes)

    def _fresh_recent_hand_boxes(
        self,
        observed_at: float,
    ) -> tuple[tuple[float, float, float, float], ...]:
        """Reuse the newest live hand boxes for brief live-stream misses only."""

        if not self._recent_hand_boxes or self._recent_hand_boxes_seen_at is None:
            return ()
        if (observed_at - self._recent_hand_boxes_seen_at) > _RECENT_HAND_BOXES_TTL_S:
            return ()
        return tuple(self._recent_hand_boxes)

    def _ensure_hand_landmark_worker(self) -> MediaPipeHandLandmarkWorker:
        """Reuse or create the bounded person-ROI hand-landmark worker."""

        with self._lock:
            if self._hand_landmark_worker is not None:
                return self._hand_landmark_worker
            self._hand_landmark_worker = MediaPipeHandLandmarkWorker(
                config=HandLandmarkWorkerConfig.from_config(self.config),
            )
            return self._hand_landmark_worker

    def _fresh_snapshot(
        self,
        snapshot: _RecognizerSnapshot | None,
        *,
        observed_at: float,
    ) -> _RecognizerSnapshot | None:
        """Return one recognizer snapshot only while it is still fresh."""

        if snapshot is None:
            return None
        age_s = observed_at - (snapshot.timestamp_ms / 1000.0)
        if not math.isfinite(age_s) or age_s < 0.0 or age_s > _DEFAULT_MAX_RESULT_AGE_S:
            return None
        return snapshot

    def _await_current_live_results(
        self,
        *,
        timestamp_ms: int,
        expect_custom: bool,
    ) -> tuple[float, bool, bool]:
        """Wait briefly for the current frame's async live callbacks to land."""

        wait_started_at = monotonic()
        with self._result_condition:
            builtin_ready = _snapshot_matches_timestamp(self._latest_builtin, timestamp_ms)
            custom_ready = (
                not expect_custom
                or _snapshot_matches_timestamp(self._latest_custom, timestamp_ms)
            )
            if not (builtin_ready and custom_ready):
                self._result_condition.wait_for(
                    lambda: (
                        _snapshot_matches_timestamp(self._latest_builtin, timestamp_ms)
                        and (
                            not expect_custom
                            or _snapshot_matches_timestamp(self._latest_custom, timestamp_ms)
                        )
                    ),
                    timeout=_DEFAULT_CURRENT_RESULT_WAIT_S,
                )
                builtin_ready = _snapshot_matches_timestamp(self._latest_builtin, timestamp_ms)
                custom_ready = (
                    not expect_custom
                    or _snapshot_matches_timestamp(self._latest_custom, timestamp_ms)
                )
        return monotonic() - wait_started_at, builtin_ready, custom_ready


def _snapshot_matches_timestamp(
    snapshot: _RecognizerSnapshot | None,
    timestamp_ms: int,
) -> bool:
    """Report whether one live snapshot belongs to the requested frame or newer."""

    if snapshot is None:
        return False
    return snapshot.timestamp_ms >= max(1, int(timestamp_ms))


def _count_hand_landmarks(result: object) -> int:
    """Count recognized hands from one MediaPipe gesture result."""

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
    """Return the center-x of the strongest open-palm hand when present."""

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


def _extract_hand_boxes(
    result: object,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return bounded normalized hand boxes from MediaPipe live landmarks."""

    hand_landmarks = getattr(result, "hand_landmarks", None)
    if hand_landmarks is None:
        return ()
    try:
        landmark_sets = list(hand_landmarks)
    except TypeError:
        return ()
    boxes: list[tuple[float, float, float, float]] = []
    for landmarks in landmark_sets:
        xs: list[float] = []
        ys: list[float] = []
        try:
            points = list(landmarks)
        except TypeError:
            continue
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
    hand_box: tuple[float, float, float, float],
) -> Any | None:
    """Crop one normalized hand box from the current frame when possible."""

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
    """Return one looser retry crop only when it actually adds hand context."""

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
) -> tuple[
    tuple[AICameraFineHandGesture, float | None],
    object,
    bool,
    tuple[AICameraFineHandGesture, float | None],
    object | None,
]:
    """Retry one hand-centered ROI gesture read with a looser local context crop.

    Pi forensic runs showed person-ROI hand detections that were real enough for
    MediaPipe Hand Landmarker but still yielded `none` or empty categories in
    the gesture classifier. Keep the first pass on the tighter crop, then do one
    bounded retry on a slightly wider hand-centered crop before giving up.
    """

    image = runtime_interface.build_image(runtime, frame_rgb=frame_rgb)
    primary_result = recognizer.recognize(image)
    primary_candidate = resolve_fine_hand_gesture(
        result=primary_result,
        category_map=category_map,
        min_score=min_score,
    )
    if primary_candidate[0] != AICameraFineHandGesture.NONE or context_frame_rgb is None:
        return primary_candidate, primary_result, False, (AICameraFineHandGesture.NONE, None), None
    context_image = runtime_interface.build_image(runtime, frame_rgb=context_frame_rgb)
    context_result = recognizer.recognize(context_image)
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
    builtin_raw_candidate: tuple[AICameraFineHandGesture, float | None],
    builtin_candidate: _PersonRoiGestureChoice,
    builtin_result: object,
    builtin_context_retry_used: bool,
    builtin_context_candidate: tuple[AICameraFineHandGesture, float | None],
    builtin_context_result: object | None,
    custom_raw_candidate: tuple[AICameraFineHandGesture, float | None],
    custom_candidate: _PersonRoiGestureChoice,
    custom_result: object | None,
    custom_context_retry_used: bool,
    custom_context_candidate: tuple[AICameraFineHandGesture, float | None],
    custom_context_result: object | None,
) -> dict[str, object]:
    """Return one bounded per-hand gesture debug summary."""

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
    """Prefer the full-frame landmark crop without relying on array truthiness."""

    gesture_frame = getattr(detection, "gesture_frame_rgb", None)
    if gesture_frame is not None:
        return gesture_frame
    return getattr(detection, "roi_frame_rgb", None)


def _resolve_detection_gesture_frame_source(detection: object) -> str:
    """Report whether gesture classification used a hand-localized full-frame crop."""

    if getattr(detection, "gesture_frame_rgb", None) is not None:
        return "full_frame_landmark_crop"
    return "roi_local_crop"


def _live_custom_gesture_enabled(config: MediaPipeVisionConfig) -> bool:
    """Return whether the live lane should load the custom recognizer at all."""

    return bool(getattr(config, "custom_gesture_model_path", None)) and bool(_LIVE_CUSTOM_FINE_GESTURE_MAP)


def _live_custom_min_score(config: MediaPipeVisionConfig) -> float:
    """Clamp live custom-gesture acceptance to a stricter HCI-specific floor."""

    configured = _coerce_unit_interval(getattr(config, "custom_gesture_min_score", None))
    return max(_LIVE_CUSTOM_MIN_SCORE_FLOOR, configured or 0.0)


def _fresh_live_results_confirm_no_hand(
    *,
    builtin_ready: bool,
    custom_ready: bool | None,
    hand_count: int,
    live_hand_box_count: int,
) -> bool:
    """Report when fresh live callbacks explicitly found no hand evidence.

    When the dedicated live-stream recognizers are current for this frame and
    still produce zero hands/boxes, broader person/full-frame rescues become
    much more likely to hallucinate than to help. The low-latency HCI lane
    should trust that fresh `no hand` signal and fail closed.
    """

    if not builtin_ready:
        return False
    if custom_ready is not True:
        return False
    return hand_count <= 0 and live_hand_box_count <= 0


def _allow_person_roi_rescue_despite_live_no_hand(
    gesture: AICameraFineHandGesture,
) -> bool:
    """Return whether one person-ROI symbol may override the live no-hand veto.

    Fresh live no-hand evidence should still block the noisy thumb rescue path.
    Keep one narrow exception for peace_sign, because accepted Pi baseline runs
    regularly recovered deliberate Victory gestures from person-ROI scans even
    when the live hand callbacks briefly dropped to zero boxes.
    """

    return gesture in _LIVE_NO_HAND_PERSON_ROI_ALLOWLIST


def _summarize_gesture_categories(
    result: object | None,
    *,
    limit: int = 3,
) -> tuple[dict[str, object], ...]:
    """Return the top raw gesture categories for bounded debug snapshots."""

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
    """Return one normalized frame shape for bounded debug output."""

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
    effective_hand_boxes: tuple[tuple[float, float, float, float], ...],
    person_roi_detection_count: int,
    live_roi_hand_box_count: int,
    live_roi_combined_gesture: str,
) -> str | None:
    """Explain when the bounded whole-frame hand rescue is still justified."""

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
    return None


def _coerce_roi_source_value(value: object) -> str:
    """Normalize one ROI source enum/string into a compact comparable token."""

    raw = getattr(value, "value", None) or str(value or "")
    return str(raw).strip()


def _person_roi_source_priority(roi_source: str) -> int:
    """Return the relative trust rank for one ROI source."""

    return int(_PERSON_ROI_SOURCE_PRIORITY.get(roi_source, 0))


def _person_roi_source_min_confidence(roi_source: str) -> float | None:
    """Return the extra gesture floor for one known person-ROI source."""

    value = _PERSON_ROI_SOURCE_MIN_CONFIDENCE.get(roi_source)
    return _coerce_unit_interval(value) if value is not None else None


def _effective_person_roi_source_min_confidence(
    *,
    roi_source: str,
    gesture_frame_source: str,
) -> float | None:
    """Return one extra person-ROI floor only while the classifier still sees the ROI crop.

    The broader person-ROI floors were added to block torso-wide false positives.
    Once the hand-landmark worker already re-cropped the gesture classifier onto
    a hand-localized full-frame crop, that body-ROI floor becomes over-aggressive
    and suppresses valid thumb gestures on the real Pi.
    """

    if gesture_frame_source == "full_frame_landmark_crop":
        return None
    return _person_roi_source_min_confidence(roi_source)


def _person_roi_short_circuit_min_confidence(config: MediaPipeVisionConfig) -> float:
    """Return the minimum score that justifies skipping later visible-person ROIs.

    MediaPipe's live-stream lane already uses tracking to keep the hot path
    low-latency. Once Twinr falls back into IMAGE-mode person-ROI rescans, the
    main latency driver becomes per-person serial hand reclassification. If the
    pose-hinted primary ROI already produced a supported gesture above the
    active model floors, later ROI rescans only add delay for the HDMI HCI path.
    """

    builtin_min_score = _coerce_unit_interval(getattr(config, "builtin_gesture_min_score", None)) or 0.0
    custom_min_score = _live_custom_min_score(config) if _live_custom_gesture_enabled(config) else 0.0
    return max(builtin_min_score, custom_min_score, _LIVE_CUSTOM_MIN_SCORE_FLOOR)


def _should_short_circuit_person_roi_scan(
    *,
    config: MediaPipeVisionConfig,
    pose_hint_attached: bool,
    candidate: _PersonRoiGestureChoice,
) -> bool:
    """Return whether the primary pose-hinted ROI is strong enough to stop scanning.

    Only the ROI that still matches the tracked primary person receives sparse
    pose hints. When that ROI already resolves one supported gesture above the
    active model floors, continuing through every additional visible person adds
    substantial Pi latency while rarely changing the outcome.
    """

    if not pose_hint_attached or candidate.gesture == AICameraFineHandGesture.NONE:
        return False
    confidence = _coerce_unit_interval(candidate.confidence)
    if confidence is None:
        return False
    return confidence >= _person_roi_short_circuit_min_confidence(config)


def _guard_person_roi_gesture_choice(
    *,
    candidate: tuple[AICameraFineHandGesture, float | None],
    roi_source: str,
    gesture_frame_source: str,
    detection_index: int,
) -> _PersonRoiGestureChoice:
    """Reject weak person-ROI symbols from broad crops before arbitration.

    Pi forensic captures showed broad torso ROIs resolving concrete thumb
    gestures on frames that did not contain a reliable hand pose. Keep that
    source-aware guard for raw ROI-local crops, but do not keep applying it
    after the hand-landmark worker already emitted a hand-localized full-frame
    gesture crop.
    """

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
    """Prefer tighter ROI sources when scores are close enough to be ambiguous."""

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
    """Merge guarded built-in/custom choices while keeping the winning source."""

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
    """Return the bounded score for one open-palm category candidate."""

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
    """Return one normalized hand center from MediaPipe landmarks."""

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
    """Return one finite normalized coordinate or ``None`` when malformed."""

    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def _round_optional_confidence(value: float | None) -> float | None:
    """Round one optional gesture/debug ratio into a compact bounded float."""

    if value is None or not math.isfinite(value):
        return None
    return round(max(0.0, min(1.0, float(value))), 3)


def _merge_hand_boxes(
    builtin_boxes: tuple[tuple[float, float, float, float], ...],
    custom_boxes: tuple[tuple[float, float, float, float], ...],
) -> tuple[tuple[float, float, float, float], ...]:
    """Merge built-in/custom hand boxes while collapsing exact duplicates."""

    merged: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for box in tuple(builtin_boxes or ()) + tuple(custom_boxes or ()):
        normalized = (
            round(float(box[0]), 4),
            round(float(box[1]), 4),
            round(float(box[2]), 4),
            round(float(box[3]), 4),
        )
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(box)
    return tuple(merged)


def _coerce_nonnegative_int(value: object, *, default: int = 0) -> int:
    """Return one bounded non-negative integer from loose debug payloads."""

    try:
        numeric = int(cast(Any, value))
    except (TypeError, ValueError):
        return default
    return max(0, numeric)


__all__ = [
    "LiveGestureFrameObservation",
    "LiveGesturePipeline",
]
