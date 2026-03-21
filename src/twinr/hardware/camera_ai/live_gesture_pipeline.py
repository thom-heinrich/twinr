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

from .config import MediaPipeVisionConfig
from .fine_hand_gestures import (
    BUILTIN_FINE_GESTURE_MAP,
    CUSTOM_FINE_GESTURE_MAP,
    combine_builtin_and_custom_gesture_choice,
    resolve_fine_hand_gesture,
)
from .mediapipe_runtime import MediaPipeTaskRuntime
from .models import AICameraFineHandGesture, AICameraGestureEvent


_DEFAULT_MAX_RESULT_AGE_S = 0.75
_WAVE_WINDOW_S = 0.55
_WAVE_MIN_SAMPLES = 3
_WAVE_MIN_SPAN_X = 0.10
_WAVE_MIN_TRAVEL_X = 0.18
_WAVE_DIRECTION_DELTA = 0.015


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

    def close(self) -> None:
        """Close live-stream recognizers and reset cached callback state."""

        with self._lock:
            self._latest_builtin = None
            self._latest_custom = None
            self._wave_tracker = _WaveGestureTracker()
        self._runtime.close()

    def observe(
        self,
        *,
        frame_rgb: Any,
        observed_at: float,
    ) -> LiveGestureFrameObservation:
        """Submit one RGB frame and return the freshest bounded live result."""

        runtime = self._runtime.load_runtime()
        image = self._runtime.build_image(runtime, frame_rgb=frame_rgb)
        timestamp_ms = self._runtime.timestamp_ms(observed_at)
        builtin_recognizer = self._runtime.ensure_live_gesture_recognizer(
            runtime,
            result_callback=self._handle_builtin_result,
        )
        builtin_recognizer.recognize_async(image, timestamp_ms)
        if self.config.custom_gesture_model_path:
            custom_recognizer = self._runtime.ensure_live_custom_gesture_recognizer(
                runtime,
                result_callback=self._handle_custom_result,
            )
            custom_recognizer.recognize_async(image, timestamp_ms)
        return self._build_observation(observed_at=observed_at)

    def _build_observation(self, *, observed_at: float) -> LiveGestureFrameObservation:
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
            fine_hand_gesture, fine_hand_confidence = combine_builtin_and_custom_gesture_choice(
                builtin_choice,
                custom_choice,
            )
            hand_count = max(
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
        result_age_s = None
        if freshest_timestamp_ms > 0:
            result_age_s = max(0.0, observed_at - (freshest_timestamp_ms / 1000.0))
        return LiveGestureFrameObservation(
            observed_at=observed_at,
            fine_hand_gesture=fine_hand_gesture,
            fine_hand_gesture_confidence=fine_hand_confidence,
            gesture_event=gesture_event,
            gesture_confidence=gesture_confidence,
            hand_count=hand_count,
            result_age_s=result_age_s,
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
        )
        with self._lock:
            self._latest_custom = snapshot

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


__all__ = [
    "LiveGestureFrameObservation",
    "LiveGesturePipeline",
]
