"""Run a dedicated low-latency MediaPipe live-stream gesture lane.

Twinr's user-facing HDMI gesture acknowledgements should not depend on the
general social-vision pipeline that also performs person detection, pose
classification, and temporal room reasoning. This module keeps a much thinner
hot path for the gestures that must feel instant to a person standing in front
of the Pi: capture one RGB frame, feed it into MediaPipe's live-stream gesture
recognizers, keep the newest callback result, and expose a compact
`LiveGestureFrameObservation` for downstream acknowledgement logic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any
import math

from twinr.hardware.hand_landmarks import HandLandmarkWorkerConfig, MediaPipeHandLandmarkWorker

from .config import MediaPipeVisionConfig
from .fine_hand_gestures import (
    BUILTIN_FINE_GESTURE_MAP,
    CUSTOM_FINE_GESTURE_MAP,
    combine_builtin_and_custom_gesture_choice,
    prefer_gesture_choice,
    resolve_fine_hand_gesture,
)
from .mediapipe_runtime import MediaPipeTaskRuntime
from .models import AICameraBox, AICameraFineHandGesture, AICameraGestureEvent


_DEFAULT_MAX_RESULT_AGE_S = 0.75
_WAVE_WINDOW_S = 0.55
_WAVE_MIN_SAMPLES = 3
_WAVE_MIN_SPAN_X = 0.10
_WAVE_MIN_TRAVEL_X = 0.18
_WAVE_DIRECTION_DELTA = 0.015
_RECENT_PRIMARY_PERSON_BOX_TTL_S = 0.45
_RECENT_VISIBLE_PERSON_BOXES_TTL_S = 0.45
_RECENT_HAND_BOXES_TTL_S = 0.45
_LIVE_GESTURE_NUM_HANDS = 1


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
    ) -> LiveGestureFrameObservation:
        """Submit one RGB frame and return the freshest bounded live result."""

        runtime = self._runtime.load_runtime()
        image = self._runtime.build_image(runtime, frame_rgb=frame_rgb)
        timestamp_ms = self._runtime.timestamp_ms(observed_at)
        builtin_recognizer = self._runtime.ensure_live_gesture_recognizer(
            runtime,
            result_callback=self._handle_builtin_result,
            num_hands_override=_LIVE_GESTURE_NUM_HANDS,
        )
        builtin_recognizer.recognize_async(image, timestamp_ms)
        return self._build_observation(
            runtime=runtime,
            frame_rgb=frame_rgb,
            observed_at=observed_at,
            timestamp_ms=timestamp_ms,
            primary_person_box=primary_person_box,
            visible_person_boxes=visible_person_boxes,
            person_count=person_count,
        )

    def _build_observation(
        self,
        *,
        runtime: dict[str, Any],
        frame_rgb: Any,
        observed_at: float,
        timestamp_ms: int,
        primary_person_box: AICameraBox | None,
        visible_person_boxes: tuple[AICameraBox, ...],
        person_count: int,
    ) -> LiveGestureFrameObservation:
        """Materialize one bounded observation from the newest callback results."""

        with self._lock:
            builtin = self._fresh_snapshot(self._latest_builtin, observed_at=observed_at)
            custom = None
            builtin_choice = (
                (AICameraFineHandGesture.NONE, None)
                if builtin is None
                else (builtin.gesture, builtin.confidence)
            )
            custom_choice = (AICameraFineHandGesture.NONE, None)
            fine_hand_gesture, fine_hand_confidence = combine_builtin_and_custom_gesture_choice(
                builtin_choice,
                custom_choice,
            )
            hand_count = max(0, 0 if builtin is None else builtin.hand_count)
            open_palm_center_x = None
            if builtin is not None and builtin.gesture == AICameraFineHandGesture.OPEN_PALM:
                open_palm_center_x = builtin.open_palm_center_x
            gesture_event, gesture_confidence = self._wave_tracker.observe(
                observed_at=observed_at,
                open_palm_center_x=open_palm_center_x,
            )
            freshest_timestamp_ms = 0 if builtin is None else builtin.timestamp_ms
            current_hand_boxes = builtin.hand_boxes if builtin is not None and builtin.hand_boxes else ()
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
        if fine_hand_gesture == AICameraFineHandGesture.NONE:
            debug_snapshot = {
                "resolved_source": "none",
                "live_builtin_gesture": builtin_choice[0].value,
                "live_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_custom_gesture": custom_choice[0].value,
                "live_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "live_custom_enabled": False,
                "live_hand_count": hand_count,
                "live_hand_box_count": len(current_hand_boxes),
                "effective_live_hand_box_count": len(effective_hand_boxes),
                "live_hand_box_source": hand_box_source,
                "live_result_age_s": _round_optional_confidence(result_age_s),
                "input_person_count": max(0, int(person_count)),
                "primary_person_box_available": primary_person_box is not None,
                "effective_primary_person_box_available": effective_primary_person_box is not None,
                "primary_person_box_source": primary_person_box_source,
                "visible_person_box_count": len(live_visible_person_boxes),
                "effective_visible_person_box_count": len(effective_visible_person_boxes),
                "visible_person_box_source": visible_person_box_source,
            }
            person_roi_choice = (AICameraFineHandGesture.NONE, None)
            if effective_visible_person_boxes:
                person_roi_choice, person_roi_debug = self._recognize_from_person_rois(
                    runtime=runtime,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    person_boxes=effective_visible_person_boxes,
                )
                debug_snapshot.update(person_roi_debug)
                if person_roi_choice[0] != AICameraFineHandGesture.NONE:
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
        else:
            debug_snapshot = {
                "resolved_source": "live_stream",
                "live_builtin_gesture": builtin_choice[0].value,
                "live_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_custom_gesture": custom_choice[0].value,
                "live_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "live_custom_enabled": False,
                "live_hand_count": hand_count,
                "live_hand_box_count": len(current_hand_boxes),
                "live_result_age_s": _round_optional_confidence(result_age_s),
                "input_person_count": max(0, int(person_count)),
                "primary_person_box_available": primary_person_box is not None,
                "effective_primary_person_box_available": effective_primary_person_box is not None,
                "primary_person_box_source": primary_person_box_source,
                "visible_person_box_count": len(live_visible_person_boxes),
                "effective_visible_person_box_count": len(effective_visible_person_boxes),
                "visible_person_box_source": visible_person_box_source,
            }
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
        person_boxes: tuple[AICameraBox, ...],
    ) -> tuple[tuple[AICameraFineHandGesture, float | None], dict[str, object]]:
        """Recover concrete symbols from bounded visible-person upper-body ROIs."""

        debug = {
            "person_roi_candidate_count": len(person_boxes),
            "person_roi_detection_count": 0,
            "person_roi_match_index": None,
            "person_roi_builtin_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_builtin_confidence": None,
            "person_roi_custom_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_custom_confidence": None,
            "person_roi_combined_gesture": AICameraFineHandGesture.NONE.value,
            "person_roi_combined_confidence": None,
        }
        best_builtin_choice = (AICameraFineHandGesture.NONE, None)
        best_custom_choice = (AICameraFineHandGesture.NONE, None)
        best_combined_choice = (AICameraFineHandGesture.NONE, None)
        best_match_index: int | None = None
        for index, person_box in enumerate(person_boxes):
            hand_landmark_result = self._ensure_hand_landmark_worker().analyze(
                runtime=runtime,
                frame_rgb=frame_rgb,
                timestamp_ms=timestamp_ms + index,
                primary_person_box=person_box,
                sparse_keypoints={},
            )
            detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
            debug["person_roi_detection_count"] = int(debug["person_roi_detection_count"]) + len(detections)
            builtin_choice, custom_choice = self._recognize_from_hand_landmark_result(
                runtime=runtime,
                hand_landmark_result=hand_landmark_result,
            )
            combined = combine_builtin_and_custom_gesture_choice(builtin_choice, custom_choice)
            if prefer_gesture_choice(best_combined_choice, combined) != best_combined_choice:
                best_builtin_choice = builtin_choice
                best_custom_choice = custom_choice
                best_combined_choice = combined
                best_match_index = index
        debug.update(
            {
                "person_roi_match_index": best_match_index,
                "person_roi_builtin_gesture": best_builtin_choice[0].value,
                "person_roi_builtin_confidence": _round_optional_confidence(best_builtin_choice[1]),
                "person_roi_custom_gesture": best_custom_choice[0].value,
                "person_roi_custom_confidence": _round_optional_confidence(best_custom_choice[1]),
                "person_roi_combined_gesture": best_combined_choice[0].value,
                "person_roi_combined_confidence": _round_optional_confidence(best_combined_choice[1]),
            }
        )
        return best_combined_choice, debug

    def _recognize_from_hand_landmark_result(
        self,
        *,
        runtime: dict[str, Any],
        hand_landmark_result: object,
    ) -> tuple[
        tuple[AICameraFineHandGesture, float | None],
        tuple[AICameraFineHandGesture, float | None],
    ]:
        """Run bounded image-mode gesture recognition on hand-ROI landmark crops."""

        detections = tuple(getattr(hand_landmark_result, "detections", ()) or ())
        if not detections:
            return (AICameraFineHandGesture.NONE, None), (AICameraFineHandGesture.NONE, None)

        builtin_choice = (AICameraFineHandGesture.NONE, None)
        custom_choice = (AICameraFineHandGesture.NONE, None)
        builtin_recognizer = self._runtime.ensure_roi_gesture_recognizer(runtime)
        custom_recognizer = None
        if self.config.custom_gesture_model_path:
            custom_recognizer = self._runtime.ensure_custom_roi_gesture_recognizer(runtime)

        for detection in detections:
            roi_frame = getattr(detection, "roi_frame_rgb", None)
            if roi_frame is None:
                continue
            roi_image = self._runtime.build_image(runtime, frame_rgb=roi_frame)
            builtin_choice = prefer_gesture_choice(
                builtin_choice,
                resolve_fine_hand_gesture(
                    result=builtin_recognizer.recognize(roi_image),
                    category_map=BUILTIN_FINE_GESTURE_MAP,
                    min_score=self.config.builtin_gesture_min_score,
                ),
            )
            if custom_recognizer is None:
                continue
            custom_choice = prefer_gesture_choice(
                custom_choice,
                resolve_fine_hand_gesture(
                    result=custom_recognizer.recognize(roi_image),
                    category_map=CUSTOM_FINE_GESTURE_MAP,
                    min_score=self.config.custom_gesture_min_score,
                ),
            )
        return builtin_choice, custom_choice

    def _handle_builtin_result(
        self,
        result: object,
        _output_image: object,
        timestamp_ms: int,
    ) -> None:
        """Store one built-in live-stream recognizer callback result."""

        gesture, confidence = resolve_fine_hand_gesture(
            result=result,
            category_map=BUILTIN_FINE_GESTURE_MAP,
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
        with self._lock:
            self._latest_builtin = snapshot

    def _handle_custom_result(
        self,
        result: object,
        _output_image: object,
        timestamp_ms: int,
    ) -> None:
        """Store one custom live-stream recognizer callback result."""

        gesture, confidence = resolve_fine_hand_gesture(
            result=result,
            category_map=CUSTOM_FINE_GESTURE_MAP,
            min_score=self.config.custom_gesture_min_score,
        )
        snapshot = _RecognizerSnapshot(
            timestamp_ms=max(1, int(timestamp_ms)),
            gesture=gesture,
            confidence=confidence,
            hand_count=_count_hand_landmarks(result),
            open_palm_center_x=None,
            hand_boxes=_extract_hand_boxes(result),
        )
        with self._lock:
            self._latest_custom = snapshot

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
        if self.config.custom_gesture_model_path:
            custom_recognizer = self._runtime.ensure_custom_roi_gesture_recognizer(runtime)

        builtin_choice = (AICameraFineHandGesture.NONE, None)
        custom_choice = (AICameraFineHandGesture.NONE, None)
        for hand_box in hand_boxes:
            crop = _crop_hand_box(frame_rgb, hand_box)
            if crop is None:
                continue
            image = self._runtime.build_image(runtime, frame_rgb=crop)
            builtin_choice = prefer_gesture_choice(
                builtin_choice,
                resolve_fine_hand_gesture(
                    result=builtin_recognizer.recognize(image),
                    category_map=BUILTIN_FINE_GESTURE_MAP,
                    min_score=self.config.builtin_gesture_min_score,
                ),
            )
            if custom_recognizer is None:
                continue
            custom_choice = prefer_gesture_choice(
                custom_choice,
                resolve_fine_hand_gesture(
                    result=custom_recognizer.recognize(image),
                    category_map=CUSTOM_FINE_GESTURE_MAP,
                    min_score=self.config.custom_gesture_min_score,
                ),
            )
        combined = combine_builtin_and_custom_gesture_choice(builtin_choice, custom_choice)
        debug.update(
            {
                "live_roi_builtin_gesture": builtin_choice[0].value,
                "live_roi_builtin_confidence": _round_optional_confidence(builtin_choice[1]),
                "live_roi_custom_gesture": custom_choice[0].value,
                "live_roi_custom_confidence": _round_optional_confidence(custom_choice[1]),
                "live_roi_combined_gesture": combined[0].value,
                "live_roi_combined_confidence": _round_optional_confidence(combined[1]),
            }
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
    y0 = max(0, min(frame_height - 1, int(math.floor(top * frame_height))))
    x0 = max(0, min(frame_width - 1, int(math.floor(left * frame_width))))
    y1 = max(y0 + 1, min(frame_height, int(math.ceil(bottom * frame_height))))
    x1 = max(x0 + 1, min(frame_width, int(math.ceil(right * frame_width))))
    try:
        return frame_rgb[y0:y1, x0:x1]
    except Exception:
        return None


def _open_palm_score(categories: object) -> float | None:
    """Return the bounded score for one open-palm category candidate."""

    try:
        candidates = list(categories)
    except TypeError:
        return None
    best_score = None
    for category in candidates:
        label = str(getattr(category, "category_name", "") or "").strip().lower()
        if label != "open_palm":
            continue
        try:
            score = float(getattr(category, "score", 0.0) or 0.0)
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
        points = list(landmarks)
    except TypeError:
        return None
    xs: list[float] = []
    for point in points:
        try:
            value = float(getattr(point, "x", None))
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
        numeric = float(value)
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


__all__ = [
    "LiveGestureFrameObservation",
    "LiveGesturePipeline",
]
