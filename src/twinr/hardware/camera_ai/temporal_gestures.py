"""Classify coarse arm gestures from a bounded short pose sequence."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Final

from .config import _clamp_ratio, _coerce_positive_float
from .models import AICameraGestureEvent

Point3D = tuple[float, float, float]

_MIN_REQUIRED_FRAMES: Final[int] = 2
_ASSUMED_MAX_CAMERA_FPS: Final[float] = 60.0
_ABSOLUTE_MAX_BUFFER_FRAMES: Final[int] = 512


def _coerce_finite_float(value: object) -> float | None:
    """Return one finite float or ``None`` when the input is unusable."""

    # AUDIT-FIX(#1): Reject non-finite and boolean pseudo-numbers before they reach hot-path gesture math.
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _coerce_min_frames(value: object, *, default: int = _MIN_REQUIRED_FRAMES) -> int:
    """Return one safe frame threshold for temporal classification."""

    # AUDIT-FIX(#4): Public APIs must defend against invalid, negative, and boolean thresholds.
    numeric = _coerce_finite_float(value)
    if numeric is None:
        return max(_MIN_REQUIRED_FRAMES, int(default))
    if numeric.is_integer():
        return max(_MIN_REQUIRED_FRAMES, int(numeric))
    return max(_MIN_REQUIRED_FRAMES, int(math.ceil(numeric)))


def _normalize_keypoint(value: object) -> Point3D | None:
    """Return one finite pose keypoint tuple or ``None`` when the sample is malformed."""

    if value is None:
        return None
    try:
        x_raw = value[0]
        y_raw = value[1]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        z_raw = value[2]
    except (TypeError, IndexError, KeyError):
        z_raw = 0.0
    x = _coerce_finite_float(x_raw)
    y = _coerce_finite_float(y_raw)
    z = _coerce_finite_float(z_raw)
    # AUDIT-FIX(#1): Treat malformed landmarks as absent so invalid model payloads degrade to False instead of crashing.
    if x is None or y is None or z is None:
        return None
    return (x, y, z)


def _extract_keypoint(sparse_keypoints: Mapping[int, object], index: int) -> Point3D | None:
    """Read one keypoint index from a sparse pose mapping."""

    return _normalize_keypoint(sparse_keypoints.get(index))


def _sorted_valid_frames(frames: object) -> tuple[TemporalPoseFrame, ...]:
    """Return chronologically ordered frames with finite timestamps only."""

    try:
        iterable_frames = tuple(frames)
    except TypeError:
        return ()
    # AUDIT-FIX(#2): Temporal heuristics must not trust caller ordering or non-finite timestamps.
    valid_frames = [
        normalized_frame
        for frame in iterable_frames
        for normalized_frame in (_coerce_temporal_pose_frame(frame),)
        if normalized_frame is not None and math.isfinite(normalized_frame.observed_at)
    ]
    valid_frames.sort(key=lambda frame: frame.observed_at)
    return tuple(valid_frames)


def _coerce_temporal_pose_frame(frame: object) -> TemporalPoseFrame | None:
    """Normalize one frame-like object into ``TemporalPoseFrame`` when possible."""

    if isinstance(frame, TemporalPoseFrame):
        return frame
    observed_at = _coerce_finite_float(getattr(frame, "observed_at", None))
    if observed_at is None:
        return None
    return TemporalPoseFrame(
        observed_at=observed_at,
        left_shoulder=_normalize_keypoint(getattr(frame, "left_shoulder", None)),
        right_shoulder=_normalize_keypoint(getattr(frame, "right_shoulder", None)),
        left_elbow=_normalize_keypoint(getattr(frame, "left_elbow", None)),
        right_elbow=_normalize_keypoint(getattr(frame, "right_elbow", None)),
        left_wrist=_normalize_keypoint(getattr(frame, "left_wrist", None)),
        right_wrist=_normalize_keypoint(getattr(frame, "right_wrist", None)),
        left_hip=_normalize_keypoint(getattr(frame, "left_hip", None)),
        right_hip=_normalize_keypoint(getattr(frame, "right_hip", None)),
    )


@dataclass(frozen=True, slots=True)
class TemporalPoseFrame:
    """Store one normalized pose sample for short-window gesture classification."""

    observed_at: float
    left_shoulder: Point3D | None
    right_shoulder: Point3D | None
    left_elbow: Point3D | None
    right_elbow: Point3D | None
    left_wrist: Point3D | None
    right_wrist: Point3D | None
    left_hip: Point3D | None
    right_hip: Point3D | None


class TemporalPoseGestureClassifier:
    """Classify coarse arm gestures from a short landmark sequence window."""

    def __init__(self, *, window_s: float, min_frames: int) -> None:
        """Initialize one bounded sequence classifier."""

        self.window_s = _coerce_positive_float(window_s, default=1.6)
        if not math.isfinite(self.window_s) or self.window_s <= 0.0:
            self.window_s = 1.6
        # AUDIT-FIX(#5): Avoid silent threshold weakening from bare int(...) truncation of float-ish config values.
        self.min_frames = _coerce_min_frames(min_frames, default=_MIN_REQUIRED_FRAMES)
        # AUDIT-FIX(#3): Bound the hot-path buffer even if timestamps stall or the producer floods frames.
        estimated_window_frames = int(math.ceil(self.window_s * _ASSUMED_MAX_CAMERA_FPS))
        self._max_buffer_frames = min(
            _ABSOLUTE_MAX_BUFFER_FRAMES,
            max(self.min_frames * 2, estimated_window_frames + self.min_frames + 4),
        )
        self._frames: deque[TemporalPoseFrame] = deque()

    def clear(self) -> None:
        """Discard any buffered sequence state."""

        self._frames.clear()

    def observe(
        self,
        *,
        observed_at: float,
        sparse_keypoints: Mapping[int, object],
    ) -> tuple[AICameraGestureEvent, float | None]:
        """Append one pose sample and classify the current coarse gesture."""

        observed_at_value = _coerce_finite_float(observed_at)
        # AUDIT-FIX(#1): Ignore unusable timestamps instead of corrupting prune logic with NaN/inf values.
        if observed_at_value is None:
            return AICameraGestureEvent.NONE, None
        # AUDIT-FIX(#2): Reset on clock regressions or out-of-order delivery so the temporal window stays chronological.
        if self._frames and observed_at_value < self._frames[-1].observed_at:
            self.clear()

        # AUDIT-FIX(#1): Fall back to an empty mapping when caller input is not dict-like.
        safe_keypoints = sparse_keypoints if isinstance(sparse_keypoints, Mapping) else {}
        self._frames.append(
            TemporalPoseFrame(
                observed_at=observed_at_value,
                # AUDIT-FIX(#1): Normalize every incoming landmark before it reaches gesture heuristics.
                left_shoulder=_extract_keypoint(safe_keypoints, 5),
                right_shoulder=_extract_keypoint(safe_keypoints, 6),
                left_elbow=_extract_keypoint(safe_keypoints, 7),
                right_elbow=_extract_keypoint(safe_keypoints, 8),
                left_wrist=_extract_keypoint(safe_keypoints, 9),
                right_wrist=_extract_keypoint(safe_keypoints, 10),
                left_hip=_extract_keypoint(safe_keypoints, 11),
                right_hip=_extract_keypoint(safe_keypoints, 12),
            )
        )
        self._prune(now=observed_at_value)
        # AUDIT-FIX(#3): Keep only the newest bounded subset when the producer misbehaves.
        while len(self._frames) > self._max_buffer_frames:
            self._frames.popleft()
        return classify_temporal_gesture(tuple(self._frames), min_frames=self.min_frames)

    def _prune(self, *, now: float) -> None:
        """Drop samples that are older than the configured temporal window."""

        now_value = _coerce_finite_float(now)
        # AUDIT-FIX(#2): Defensive revalidation keeps internal state recoverable even if the method is called incorrectly.
        if now_value is None:
            self.clear()
            return
        cutoff = now_value - self.window_s
        while self._frames and (
            not math.isfinite(self._frames[0].observed_at) or self._frames[0].observed_at < cutoff
        ):
            self._frames.popleft()


def classify_temporal_gesture(
    frames: tuple[TemporalPoseFrame, ...],
    *,
    min_frames: int,
) -> tuple[AICameraGestureEvent, float | None]:
    """Classify one coarse arm gesture from a short sequence of pose frames."""

    # AUDIT-FIX(#4): Sanitize the public threshold so 0, negatives, and bools cannot trigger false positives.
    effective_min_frames = _coerce_min_frames(min_frames, default=_MIN_REQUIRED_FRAMES)
    # AUDIT-FIX(#2): Public callers can bypass the class wrapper, so temporal analysis reorders frames chronologically here.
    ordered_frames = _sorted_valid_frames(frames)
    if len(ordered_frames) < effective_min_frames:
        return AICameraGestureEvent.NONE, None

    arms_crossed_matches = sum(1 for frame in ordered_frames if frame_is_arms_crossed(frame))
    timeout_matches = sum(1 for frame in ordered_frames if frame_is_timeout_t(frame))
    two_hand_dismiss_matches = sum(1 for frame in ordered_frames if frame_is_two_hand_dismiss(frame))
    frame_count = len(ordered_frames)
    if arms_crossed_matches >= effective_min_frames:
        return AICameraGestureEvent.ARMS_CROSSED, round(
            _clamp_ratio(0.58 + 0.08 * (arms_crossed_matches / frame_count), default=0.58),
            3,
        )
    if timeout_matches >= effective_min_frames:
        return AICameraGestureEvent.TIMEOUT_T, round(
            _clamp_ratio(0.58 + 0.08 * (timeout_matches / frame_count), default=0.58),
            3,
        )
    if two_hand_dismiss_matches >= effective_min_frames:
        return AICameraGestureEvent.TWO_HAND_DISMISS, round(
            _clamp_ratio(0.60 + 0.08 * (two_hand_dismiss_matches / frame_count), default=0.60),
            3,
        )
    if frames_form_wave(ordered_frames, min_frames=effective_min_frames):
        return AICameraGestureEvent.WAVE, 0.66
    return AICameraGestureEvent.NONE, None


def frame_is_arms_crossed(frame: TemporalPoseFrame) -> bool:
    """Return whether one pose frame looks like crossed arms at chest height."""

    frame = _coerce_temporal_pose_frame(frame)
    if frame is None:
        return False
    left_shoulder = frame.left_shoulder
    right_shoulder = frame.right_shoulder
    left_wrist = frame.left_wrist
    right_wrist = frame.right_wrist
    # AUDIT-FIX(#6): Use explicit guards instead of asserts so behavior stays stable under python -O.
    if left_shoulder is None or right_shoulder is None or left_wrist is None or right_wrist is None:
        return False
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    chest_bottom = shoulder_y + 0.24
    return (
        shoulder_y - 0.06 <= left_wrist[1] <= chest_bottom
        and shoulder_y - 0.06 <= right_wrist[1] <= chest_bottom
        and left_wrist[0] >= shoulder_center_x + 0.02
        and right_wrist[0] <= shoulder_center_x - 0.02
    )


def frame_is_timeout_t(frame: TemporalPoseFrame) -> bool:
    """Return whether one pose frame looks like a timeout-T arm configuration."""

    frame = _coerce_temporal_pose_frame(frame)
    if frame is None:
        return False
    variants = (
        (
            frame.left_shoulder,
            frame.left_elbow,
            frame.left_wrist,
            frame.right_shoulder,
            frame.right_elbow,
            frame.right_wrist,
        ),
        (
            frame.right_shoulder,
            frame.right_elbow,
            frame.right_wrist,
            frame.left_shoulder,
            frame.left_elbow,
            frame.left_wrist,
        ),
    )
    for (
        horizontal_shoulder,
        horizontal_elbow,
        horizontal_wrist,
        vertical_shoulder,
        vertical_elbow,
        vertical_wrist,
    ) in variants:
        # AUDIT-FIX(#6): Explicit None checks keep this heuristic stable in optimized and non-optimized runtimes.
        if (
            horizontal_shoulder is None
            or horizontal_elbow is None
            or horizontal_wrist is None
            or vertical_shoulder is None
            or vertical_elbow is None
            or vertical_wrist is None
        ):
            continue
        horizontal_ok = (
            abs(horizontal_wrist[1] - horizontal_elbow[1]) <= 0.08
            and abs(horizontal_elbow[1] - horizontal_shoulder[1]) <= 0.16
        )
        vertical_ok = (
            abs(vertical_wrist[0] - vertical_elbow[0]) <= 0.06
            and abs(vertical_elbow[0] - vertical_shoulder[0]) <= 0.06
            and vertical_wrist[1] < vertical_elbow[1] < (vertical_shoulder[1] + 0.20)
        )
        wrists_close = (
            abs(horizontal_wrist[0] - vertical_wrist[0]) <= 0.12
            and abs(horizontal_wrist[1] - vertical_elbow[1]) <= 0.12
        )
        if horizontal_ok and vertical_ok and wrists_close:
            return True
    return False


def frame_is_two_hand_dismiss(frame: TemporalPoseFrame) -> bool:
    """Return whether one frame looks like a symmetric two-hand dismiss pose."""

    frame = _coerce_temporal_pose_frame(frame)
    if frame is None:
        return False
    left_shoulder = frame.left_shoulder
    right_shoulder = frame.right_shoulder
    left_wrist = frame.left_wrist
    right_wrist = frame.right_wrist
    # AUDIT-FIX(#6): Use explicit guards instead of asserts so invalid frames simply evaluate to False.
    if left_shoulder is None or right_shoulder is None or left_wrist is None or right_wrist is None:
        return False
    return (
        left_wrist[0] <= left_shoulder[0] - 0.12
        and right_wrist[0] >= right_shoulder[0] + 0.12
        and abs(left_wrist[1] - left_shoulder[1]) <= 0.18
        and abs(right_wrist[1] - right_shoulder[1]) <= 0.18
    )


def _frames_form_wave_ordered(
    ordered_frames: tuple[TemporalPoseFrame, ...],
    *,
    min_frames: int,
) -> bool:
    """Return whether one ordered sequence contains a raised-hand lateral wave pattern."""

    for wrist_name, shoulder_name in (("left_wrist", "left_shoulder"), ("right_wrist", "right_shoulder")):
        series: list[float] = []
        for frame in ordered_frames:
            wrist = getattr(frame, wrist_name)
            shoulder = getattr(frame, shoulder_name)
            if wrist is None or shoulder is None:
                continue
            if wrist[1] >= shoulder[1] - 0.05:
                continue
            series.append(wrist[0] - shoulder[0])
        if len(series) < min_frames:
            continue
        amplitude = max(series) - min(series)
        if amplitude < 0.10:
            continue
        direction_changes = 0
        last_sign = 0
        for index in range(1, len(series)):
            delta = series[index] - series[index - 1]
            sign = 1 if delta > 0.02 else -1 if delta < -0.02 else 0
            if sign and last_sign and sign != last_sign:
                direction_changes += 1
            if sign:
                last_sign = sign
        if direction_changes >= 1:
            return True
    return False


def frames_form_wave(frames: tuple[TemporalPoseFrame, ...], *, min_frames: int) -> bool:
    """Return whether one sequence contains a raised-hand lateral wave pattern."""

    # AUDIT-FIX(#4): Direct callers should get the same safe thresholding as the main classifier.
    effective_min_frames = _coerce_min_frames(min_frames, default=_MIN_REQUIRED_FRAMES)
    # AUDIT-FIX(#2): Direct callers should also get chronological ordering before temporal wave analysis.
    ordered_frames = _sorted_valid_frames(frames)
    if len(ordered_frames) < effective_min_frames:
        return False
    return _frames_form_wave_ordered(ordered_frames, min_frames=effective_min_frames)


__all__ = [
    "TemporalPoseFrame",
    "TemporalPoseGestureClassifier",
    "classify_temporal_gesture",
    "frame_is_arms_crossed",
    "frame_is_timeout_t",
    "frame_is_two_hand_dismiss",
    "frames_form_wave",
]
