# CHANGELOG: 2026-03-28
# BUG-1: Fixed seated false positives caused by accepting knees that were not actually below the hips.
# BUG-2: Fixed upright->slumped false positives caused by letting fallback-box position override strong skeletal evidence.
# BUG-3: Fixed command/body decisions ignoring post-visibility joint confidence, which let weak landmarks trigger high-impact labels.
# BUG-4: Restored the legacy single-frame gesture contract while keeping temporal wave verification and debounce for explicit stream callers only.
# SEC-1: Added optional temporal debounce / hysteresis / cooldown hooks so one transient frame cannot inject STOP/DISMISS/CONFIRM in a live stream.
# IMP-1: Replaced brittle fixed absolute thresholds with scale-normalized geometric scoring derived from torso, shoulders, and fallback box scale.
# IMP-2: Added optional world-keypoint support so callers can use 2026 pose stacks that expose metric 3D landmarks.
# IMP-3: Added evidence scoring + abstention so ambiguous frames fail closed as NONE/UNKNOWN instead of forcing a label.

"""Classify bounded coarse body and gesture states from keypoints.

This module keeps the legacy drop-in functional API, but it also exposes an
optional temporal state object for live-stream use. When a temporal state and a
timestamp are supplied, command gestures are debounced and emitted on stable
rising edges instead of firing on every noisy frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Mapping, Sequence, TypeVar

from .config import _clamp_ratio
from .models import AICameraBodyPose, AICameraBox, AICameraGestureEvent
from .pose_features import visible_joint

Joint = tuple[float, float, float]
T = TypeVar("T")


def _finite_number(value: object, *, default: float = 0.0) -> float:
    """Return a finite float or a safe default."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sanitize_joint(joint: Joint | Sequence[object] | None) -> Joint | None:
    """Return a finite, confidence-bounded 3-value joint tuple or None."""

    if joint is None:
        return None
    try:
        if len(joint) != 3:
            return None
        x = float(joint[0])
        y = float(joint[1])
        confidence = float(joint[2])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(confidence)):
        return None
    return (x, y, _clamp01(confidence))


def _safe_visible_joint(
    keypoints: Mapping[int, Joint],
    index: int,
) -> Joint | None:
    """Return one sanitized visible joint."""

    return _sanitize_joint(visible_joint(keypoints, index))


def _raw_sanitized_joint(
    keypoints: Mapping[int, Joint] | None,
    index: int,
) -> Joint | None:
    """Return a sanitized raw joint without visibility filtering."""

    if keypoints is None:
        return None
    try:
        raw_joint = keypoints[index]
    except (KeyError, IndexError, TypeError):
        return None
    return _sanitize_joint(raw_joint)


def _box_metric(box: AICameraBox, attr: str, *, default: float = 0.0) -> float:
    """Return one finite fallback-box metric."""

    return _finite_number(getattr(box, attr, default), default=default)


def _clamp01(value: float) -> float:
    """Clamp one finite scalar into [0, 1]."""

    return min(1.0, max(0.0, _finite_number(value, default=0.0)))


def _safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Return a finite ratio or a safe default."""

    denominator = _finite_number(denominator, default=0.0)
    if abs(denominator) < 1e-9:
        return default
    return _finite_number(numerator / denominator, default=default)


def _distance(a: Joint | None, b: Joint | None) -> float:
    """Return Euclidean distance between two joints in 2D."""

    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _joint_confidence(*joints: Joint | None) -> float:
    """Return one conservative joint-confidence aggregate."""

    confs = [_clamp01(joint[2]) for joint in joints if joint is not None]
    return min(confs) if confs else 0.0


def _soft_pass_min(value: float, threshold: float, *, softness: float) -> float:
    """Return one fuzzy score for value >= threshold."""

    value = _finite_number(value, default=0.0)
    threshold = _finite_number(threshold, default=0.0)
    softness = max(1e-6, abs(_finite_number(softness, default=0.0)))
    if value >= threshold:
        return 1.0
    if value <= threshold - softness:
        return 0.0
    return (value - (threshold - softness)) / softness


def _soft_pass_max(value: float, threshold: float, *, softness: float) -> float:
    """Return one fuzzy score for value <= threshold."""

    value = _finite_number(value, default=0.0)
    threshold = _finite_number(threshold, default=0.0)
    softness = max(1e-6, abs(_finite_number(softness, default=0.0)))
    if value <= threshold:
        return 1.0
    if value >= threshold + softness:
        return 0.0
    return 1.0 - ((value - threshold) / softness)


def _soft_band(value: float, *, center: float, half_width: float, softness: float) -> float:
    """Return one fuzzy score for staying inside a center band."""

    distance = abs(_finite_number(value, default=0.0) - _finite_number(center, default=0.0))
    return _soft_pass_max(distance, max(0.0, half_width), softness=max(1e-6, softness))


def _rounded_confidence(*, score: float, attention_score: float, evidence: float = 1.0) -> float:
    """Return one safe rounded confidence from fused evidence."""

    score = _clamp01(score)
    evidence = _clamp01(evidence)
    attention_score = _clamp01(attention_score)
    fused = _clamp_ratio(
        0.08 + 0.72 * score + 0.12 * evidence + 0.08 * attention_score,
        default=score,
    )
    return round(_clamp01(fused), 3)


def _legacy_rounded_confidence(*, base: float, scale: float, attention_score: float) -> float:
    """Return the pre-refactor single-frame confidence contract."""

    bounded = _finite_number(
        _clamp_ratio(base + scale * attention_score, default=base),
        default=base,
    )
    return round(bounded, 3)


@dataclass(slots=True)
class _GeometryContext:
    """Scale context derived from the current person geometry."""

    box_top: float
    box_width: float
    box_height: float
    box_center_x: float
    box_center_y: float
    shoulder_span: float
    torso_height: float
    person_height: float
    center_shoulder_x: float
    center_shoulder_y: float

    @property
    def horizontal_scale(self) -> float:
        return max(0.05, self.shoulder_span, self.box_width * 0.40)

    @property
    def vertical_scale(self) -> float:
        return max(0.06, self.torso_height, self.box_height * 0.35)

    @property
    def arm_scale(self) -> float:
        return max(0.06, self.shoulder_span * 0.70, self.vertical_scale * 0.55)

    @property
    def body_scale(self) -> float:
        return max(0.12, self.person_height, self.box_height * 0.90)


@dataclass(slots=True)
class _BodyEvidence:
    """Intermediate body pose measurements."""

    torso_angle_from_vertical_deg: float | None = None
    torso_length: float = 0.0
    torso_confidence: float = 0.0
    knee_drop: float | None = None
    ankle_drop: float | None = None
    lower_chain_confidence: float = 0.0
    source: str = "box"


@dataclass(slots=True)
class _WristHistorySample:
    """One wrist-motion sample for temporal wave verification."""

    timestamp_ms: int
    lateral: float
    raised: bool
    confidence: float


@dataclass(slots=True)
class PoseTemporalState:
    """Optional stream state for debounced gesture/body classification."""

    gesture_candidate: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_candidate_frames: int = 0
    gesture_last_emitted: AICameraGestureEvent = AICameraGestureEvent.NONE
    gesture_cooldown_until_ms: int = 0
    body_candidate: AICameraBodyPose = AICameraBodyPose.UNKNOWN
    body_candidate_frames: int = 0
    body_stable: AICameraBodyPose = AICameraBodyPose.UNKNOWN
    left_wrist_history: Deque[_WristHistorySample] = field(default_factory=lambda: deque(maxlen=12))
    right_wrist_history: Deque[_WristHistorySample] = field(default_factory=lambda: deque(maxlen=12))


def _build_geometry_context(
    *,
    left_shoulder: Joint | None,
    right_shoulder: Joint | None,
    left_hip: Joint | None = None,
    right_hip: Joint | None = None,
    left_ankle: Joint | None = None,
    right_ankle: Joint | None = None,
    fallback_box: AICameraBox,
) -> _GeometryContext:
    """Return per-frame geometry scales from joints and fallback box."""

    box_top = _box_metric(fallback_box, "top", default=0.0)
    box_width = max(0.0, _box_metric(fallback_box, "width", default=0.0))
    box_height = max(0.0, _box_metric(fallback_box, "height", default=0.0))
    box_center_x = _box_metric(fallback_box, "center_x", default=0.0)
    box_center_y = _box_metric(fallback_box, "center_y", default=0.0)

    shoulder_span = 0.0
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_span = abs(right_shoulder[0] - left_shoulder[0])

    shoulder_center_x = box_center_x
    shoulder_center_y = box_top + box_height * 0.28 if box_height > 0.0 else 0.38
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
    elif left_shoulder is not None:
        shoulder_center_x = left_shoulder[0]
        shoulder_center_y = left_shoulder[1]
    elif right_shoulder is not None:
        shoulder_center_x = right_shoulder[0]
        shoulder_center_y = right_shoulder[1]

    torso_height = 0.0
    if left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None:
        hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
        torso_height = max(0.0, hip_center_y - shoulder_center_y)
    elif left_shoulder is not None and left_hip is not None:
        torso_height = max(0.0, left_hip[1] - left_shoulder[1])
    elif right_shoulder is not None and right_hip is not None:
        torso_height = max(0.0, right_hip[1] - right_shoulder[1])
    torso_height = max(torso_height, box_height * 0.22, shoulder_span * 0.75, 0.08)

    person_height = 0.0
    if left_shoulder is not None and right_shoulder is not None and left_ankle is not None and right_ankle is not None:
        ankle_center_y = (left_ankle[1] + right_ankle[1]) / 2.0
        person_height = max(0.0, ankle_center_y - shoulder_center_y)
    elif left_shoulder is not None and left_ankle is not None:
        person_height = max(0.0, left_ankle[1] - left_shoulder[1])
    elif right_shoulder is not None and right_ankle is not None:
        person_height = max(0.0, right_ankle[1] - right_shoulder[1])
    person_height = max(person_height, box_height, torso_height * 1.8, 0.18)

    return _GeometryContext(
        box_top=box_top,
        box_width=box_width,
        box_height=box_height,
        box_center_x=box_center_x,
        box_center_y=box_center_y,
        shoulder_span=max(0.08, shoulder_span, box_width * 0.18),
        torso_height=torso_height,
        person_height=person_height,
        center_shoulder_x=shoulder_center_x,
        center_shoulder_y=shoulder_center_y,
    )


def _arm_vertical_score(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    geom: _GeometryContext,
) -> float:
    """Return one fuzzy score for a vertical raised arm."""

    if shoulder is None or elbow is None or wrist is None:
        return 0.0
    x_tolerance = max(0.05, geom.shoulder_span * 0.18)
    raise_min = max(0.06, geom.vertical_scale * 0.24)
    elbow_below_shoulder_tol = max(0.10, geom.vertical_scale * 0.32)

    vertical_score = min(
        _soft_pass_max(abs(wrist[0] - elbow[0]), x_tolerance, softness=x_tolerance * 0.80),
        _soft_pass_max(abs(elbow[0] - shoulder[0]), x_tolerance, softness=x_tolerance * 0.80),
        _soft_pass_min(shoulder[1] - wrist[1], raise_min, softness=raise_min * 0.70),
        _soft_pass_max(elbow[1] - shoulder[1], elbow_below_shoulder_tol, softness=elbow_below_shoulder_tol * 0.80),
        _soft_pass_min(elbow[1] - wrist[1], max(0.03, geom.vertical_scale * 0.12), softness=max(0.02, geom.vertical_scale * 0.10)),
    )
    return vertical_score * _joint_confidence(shoulder, elbow, wrist)


def _arm_horizontal_toward_center_score(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    shoulder_center_x: float,
    shoulder_y: float,
    side: str,
    geom: _GeometryContext,
) -> float:
    """Return one fuzzy score for a horizontal arm toward the torso center."""

    if shoulder is None or elbow is None or wrist is None:
        return 0.0
    if side not in {"left", "right"}:
        return 0.0

    toward_center = wrist[0] > shoulder[0] if side == "left" else wrist[0] < shoulder[0]
    if not toward_center:
        return 0.0

    y_tol = max(0.06, geom.vertical_scale * 0.22)
    center_tol = max(0.08, geom.horizontal_scale * 0.26)

    score = min(
        _soft_pass_max(abs(wrist[1] - elbow[1]), y_tol, softness=y_tol * 0.90),
        _soft_pass_max(abs(elbow[1] - shoulder_y), y_tol, softness=y_tol * 0.90),
        _soft_pass_max(abs(wrist[0] - shoulder_center_x), center_tol, softness=center_tol * 0.90),
    )
    return score * _joint_confidence(shoulder, elbow, wrist)


def _wave_static_pose_score(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    side: str,
    geom: _GeometryContext,
) -> float:
    """Return one fuzzy score for a raised-wave arm pose without motion evidence."""

    if shoulder is None or elbow is None or wrist is None:
        return 0.0
    if side not in {"left", "right"}:
        return 0.0

    lateral = wrist[0] - shoulder[0]
    if side == "left":
        lateral *= -1.0

    raise_min = max(0.05, geom.vertical_scale * 0.18)
    lateral_min = max(0.10, geom.horizontal_scale * 0.32)
    elbow_low_margin = max(0.18, geom.vertical_scale * 0.52)

    pose_score = min(
        _soft_pass_min(shoulder[1] - wrist[1], raise_min, softness=raise_min * 0.70),
        _soft_pass_min(lateral, lateral_min, softness=lateral_min * 0.70),
        _soft_pass_min(elbow[1] - wrist[1], max(0.02, geom.vertical_scale * 0.08), softness=max(0.02, geom.vertical_scale * 0.08)),
        _soft_pass_max(elbow[1] - shoulder[1], elbow_low_margin, softness=elbow_low_margin * 0.80),
    )
    return pose_score * _joint_confidence(shoulder, elbow, wrist)


def _command_scores_for_arm(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    side: str,
    geom: _GeometryContext,
) -> dict[AICameraGestureEvent, float]:
    """Return fuzzy evidence scores for one arm's command-like gestures."""

    if shoulder is None or wrist is None:
        return {}
    if side not in {"left", "right"}:
        return {}

    confidences = _joint_confidence(shoulder, elbow, wrist)

    raise_min = max(0.08, geom.vertical_scale * 0.28)
    center_tol = max(0.09, geom.horizontal_scale * 0.24)
    level_tol = max(0.10, geom.vertical_scale * 0.28)
    outward_min = max(0.12, geom.horizontal_scale * 0.34)

    wrist_x_delta = wrist[0] - shoulder[0]
    outward = -wrist_x_delta if side == "left" else wrist_x_delta

    stop_score = min(
        _soft_pass_min(shoulder[1] - wrist[1], raise_min, softness=raise_min * 0.75),
        _soft_pass_max(abs(wrist_x_delta), center_tol, softness=center_tol * 0.80),
    )
    if elbow is not None:
        stop_score = min(
            stop_score,
            _soft_pass_min(elbow[1] - wrist[1], max(0.03, geom.vertical_scale * 0.12), softness=max(0.02, geom.vertical_scale * 0.10)),
        )

    dismiss_score = min(
        _soft_pass_min(outward, outward_min, softness=outward_min * 0.75),
        _soft_pass_max(abs(wrist[1] - shoulder[1]), level_tol, softness=level_tol * 0.80),
    )

    confirm_score = 0.0
    if elbow is not None:
        confirm_score = min(
            _soft_pass_max(abs(wrist_x_delta), max(0.07, geom.horizontal_scale * 0.16), softness=max(0.05, geom.horizontal_scale * 0.14)),
            _soft_pass_min(shoulder[1] - wrist[1], max(0.01, geom.vertical_scale * 0.04), softness=max(0.03, geom.vertical_scale * 0.12)),
            _soft_pass_min(elbow[1] - wrist[1], max(0.04, geom.vertical_scale * 0.14), softness=max(0.03, geom.vertical_scale * 0.10)),
        )

    return {
        AICameraGestureEvent.STOP: stop_score * confidences,
        AICameraGestureEvent.DISMISS: dismiss_score * confidences,
        AICameraGestureEvent.CONFIRM: confirm_score * confidences,
    }


def _update_wrist_history(
    *,
    state: PoseTemporalState | None,
    timestamp_ms: int | None,
    side: str,
    shoulder: Joint | None,
    wrist: Joint | None,
    geom: _GeometryContext,
) -> None:
    """Update one side's wrist history for wave detection."""

    if state is None or timestamp_ms is None or shoulder is None or wrist is None:
        return
    history = state.left_wrist_history if side == "left" else state.right_wrist_history
    lateral = wrist[0] - shoulder[0]
    if side == "left":
        lateral *= -1.0
    history.append(
        _WristHistorySample(
            timestamp_ms=timestamp_ms,
            lateral=lateral,
            raised=wrist[1] < shoulder[1] - max(0.03, geom.vertical_scale * 0.10),
            confidence=_joint_confidence(shoulder, wrist),
        )
    )


def _wave_motion_score(
    *,
    state: PoseTemporalState | None,
    timestamp_ms: int | None,
    side: str,
    geom: _GeometryContext,
) -> float:
    """Return one temporal wave-motion score from recent wrist history."""

    if state is None or timestamp_ms is None:
        return 0.0
    history = state.left_wrist_history if side == "left" else state.right_wrist_history
    window_ms = 900
    recent = [
        sample
        for sample in history
        if 0 <= timestamp_ms - sample.timestamp_ms <= window_ms and sample.raised and sample.confidence >= 0.20
    ]
    if len(recent) < 3:
        return 0.0

    laterals = [sample.lateral for sample in recent]
    diffs = [b - a for a, b in zip(laterals, laterals[1:])]
    significant_diffs = [diff for diff in diffs if abs(diff) >= max(0.015, geom.horizontal_scale * 0.05)]
    sign_changes = sum(
        1
        for prev, curr in zip(significant_diffs, significant_diffs[1:])
        if (prev < 0.0 < curr) or (prev > 0.0 > curr)
    )
    amplitude = max(laterals) - min(laterals)
    amplitude_min = max(0.06, geom.horizontal_scale * 0.18)

    amplitude_score = _soft_pass_min(amplitude, amplitude_min, softness=amplitude_min * 0.70)
    alternation_score = _soft_pass_min(float(sign_changes), 1.0, softness=1.0)
    conf_score = min(sample.confidence for sample in recent)
    return amplitude_score * alternation_score * conf_score


def _select_best_event(
    candidates: Mapping[AICameraGestureEvent, float],
) -> tuple[AICameraGestureEvent, float]:
    """Return the highest-confidence gesture candidate with abstention."""

    if not candidates:
        return AICameraGestureEvent.NONE, 0.0

    priority = {
        AICameraGestureEvent.TIMEOUT_T: 6,
        AICameraGestureEvent.TWO_HAND_DISMISS: 6,
        AICameraGestureEvent.STOP: 5,
        AICameraGestureEvent.DISMISS: 5,
        AICameraGestureEvent.CONFIRM: 5,
        AICameraGestureEvent.ARMS_CROSSED: 4,
        AICameraGestureEvent.WAVE: 3,
        AICameraGestureEvent.NONE: 0,
    }
    ranked = sorted(
        candidates.items(),
        key=lambda item: (item[1], priority.get(item[0], 0)),
        reverse=True,
    )
    best_event, best_score = ranked[0]
    if best_score < 0.52:
        return AICameraGestureEvent.NONE, 0.0
    if len(ranked) >= 2:
        _, second_score = ranked[1]
        if second_score >= best_score - 0.05 and best_score < 0.80:
            return AICameraGestureEvent.NONE, 0.0
    return best_event, _clamp01(best_score)


def _required_gesture_frames(event: AICameraGestureEvent, score: float) -> int:
    """Return one event-specific debounce length."""

    if event is AICameraGestureEvent.WAVE:
        return 3
    return 2


def _debounce_gesture_event(
    *,
    event: AICameraGestureEvent,
    score: float,
    state: PoseTemporalState | None,
    timestamp_ms: int | None,
) -> tuple[AICameraGestureEvent, float]:
    """Return a debounced gesture event for live streams."""

    if state is None or timestamp_ms is None:
        return event, score
    if event is AICameraGestureEvent.NONE:
        state.gesture_candidate = AICameraGestureEvent.NONE
        state.gesture_candidate_frames = 0
        if timestamp_ms >= state.gesture_cooldown_until_ms:
            state.gesture_last_emitted = AICameraGestureEvent.NONE
        return AICameraGestureEvent.NONE, 0.0

    if event == state.gesture_candidate:
        state.gesture_candidate_frames += 1
    else:
        state.gesture_candidate = event
        state.gesture_candidate_frames = 1

    if state.gesture_candidate_frames < _required_gesture_frames(event, score):
        return AICameraGestureEvent.NONE, 0.0

    if event == state.gesture_last_emitted and timestamp_ms < state.gesture_cooldown_until_ms:
        return AICameraGestureEvent.NONE, 0.0

    cooldown_ms = 1100 if event is AICameraGestureEvent.WAVE else 750
    state.gesture_last_emitted = event
    state.gesture_cooldown_until_ms = timestamp_ms + cooldown_ms
    return event, score


def matches_wave_arm(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    side: str,
) -> bool:
    """Return whether one arm geometry looks like a conservative raised-wave pose."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    geom = _GeometryContext(
        box_top=0.0,
        box_width=max(0.0, abs(wrist[0] - shoulder[0]) * 2.0),
        box_height=max(0.0, abs(wrist[1] - shoulder[1]) * 2.0),
        box_center_x=(shoulder[0] + wrist[0]) / 2.0,
        box_center_y=(shoulder[1] + wrist[1]) / 2.0,
        shoulder_span=max(0.08, abs(wrist[0] - shoulder[0]) * 1.2),
        torso_height=max(0.10, abs(wrist[1] - shoulder[1]) * 1.3),
        person_height=max(0.18, abs(wrist[1] - shoulder[1]) * 3.0),
        center_shoulder_x=shoulder[0],
        center_shoulder_y=shoulder[1],
    )
    return _wave_static_pose_score(
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        side=side,
        geom=geom,
    ) >= 0.62


def matches_vertical_arm(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    shoulder_span: float,
) -> bool:
    """Return whether one arm is held mostly vertical."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    geom = _GeometryContext(
        box_top=0.0,
        box_width=max(0.0, shoulder_span * 1.6),
        box_height=max(0.0, abs(wrist[1] - shoulder[1]) * 2.2),
        box_center_x=shoulder[0],
        box_center_y=(shoulder[1] + wrist[1]) / 2.0,
        shoulder_span=max(0.08, shoulder_span),
        torso_height=max(0.10, abs(wrist[1] - shoulder[1]) * 1.5),
        person_height=max(0.20, abs(wrist[1] - shoulder[1]) * 3.0),
        center_shoulder_x=shoulder[0],
        center_shoulder_y=shoulder[1],
    )
    return _arm_vertical_score(shoulder=shoulder, elbow=elbow, wrist=wrist, geom=geom) >= 0.66


def matches_horizontal_arm_toward_center(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    shoulder_center_x: float,
    shoulder_y: float,
    side: str,
) -> bool:
    """Return whether one forearm is held roughly horizontal toward the torso center."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    geom = _GeometryContext(
        box_top=0.0,
        box_width=max(0.0, abs(wrist[0] - shoulder[0]) * 2.0),
        box_height=max(0.0, abs(wrist[1] - shoulder[1]) * 2.0),
        box_center_x=shoulder_center_x,
        box_center_y=shoulder_y,
        shoulder_span=max(0.08, abs(shoulder_center_x - shoulder[0]) * 2.0),
        torso_height=max(0.10, abs(wrist[1] - shoulder[1]) * 1.5),
        person_height=max(0.20, abs(wrist[1] - shoulder[1]) * 3.0),
        center_shoulder_x=shoulder_center_x,
        center_shoulder_y=shoulder_y,
    )
    return _arm_horizontal_toward_center_score(
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
        shoulder_center_x=shoulder_center_x,
        shoulder_y=shoulder_y,
        side=side,
        geom=geom,
    ) >= 0.66


def _legacy_classify_single_arm_command(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    side: str,
    attention_score: float,
) -> tuple[AICameraGestureEvent, float] | None:
    """Return the legacy explicit single-arm command gesture if present."""

    if shoulder is None or wrist is None:
        return None
    if side not in {"left", "right"}:
        return None
    if wrist[1] < shoulder[1] - 0.08 and abs(wrist[0] - shoulder[0]) <= 0.16:
        return AICameraGestureEvent.STOP, _legacy_rounded_confidence(
            base=0.7,
            scale=0.3,
            attention_score=attention_score,
        )
    outward = wrist[0] < shoulder[0] - 0.18 if side == "left" else wrist[0] > shoulder[0] + 0.18
    if outward and abs(wrist[1] - shoulder[1]) <= 0.16:
        return AICameraGestureEvent.DISMISS, _legacy_rounded_confidence(
            base=0.6,
            scale=0.2,
            attention_score=attention_score,
        )
    if (
        elbow is not None
        and abs(wrist[0] - shoulder[0]) <= 0.08
        and wrist[1] <= shoulder[1]
        and elbow[1] > wrist[1]
    ):
        return AICameraGestureEvent.CONFIRM, _legacy_rounded_confidence(
            base=0.55,
            scale=0.2,
            attention_score=attention_score,
        )
    return None


def _legacy_matches_wave_arm(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    side: str,
) -> bool:
    """Return whether one arm matches the historic single-frame wave pose."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    if side not in {"left", "right"}:
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


def _legacy_matches_vertical_arm(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    shoulder_span: float,
) -> bool:
    """Return whether one arm matches the historic vertical-arm timeout pose."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    x_tolerance = max(0.05, shoulder_span * 0.18)
    return (
        abs(wrist[0] - elbow[0]) <= x_tolerance
        and abs(elbow[0] - shoulder[0]) <= x_tolerance
        and wrist[1] < elbow[1]
        and wrist[1] <= shoulder[1] - 0.06
        and elbow[1] <= shoulder[1] + 0.10
    )


def _legacy_matches_horizontal_arm_toward_center(
    *,
    shoulder: Joint | None,
    elbow: Joint | None,
    wrist: Joint | None,
    shoulder_center_x: float,
    shoulder_y: float,
    side: str,
) -> bool:
    """Return whether one forearm matches the historic timeout-T crossbar pose."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    if side not in {"left", "right"}:
        return False
    toward_center = wrist[0] > shoulder[0] if side == "left" else wrist[0] < shoulder[0]
    return (
        toward_center
        and abs(wrist[1] - elbow[1]) <= 0.08
        and abs(elbow[1] - shoulder_y) <= 0.16
        and abs(wrist[0] - shoulder_center_x) <= 0.16
    )


def _classify_gesture_single_frame_legacy(
    *,
    keypoints: Mapping[int, Joint],
    attention_score: float,
    fallback_box: AICameraBox,
) -> tuple[AICameraGestureEvent, float | None]:
    """Return the historic single-frame gesture contract."""

    left_shoulder = _safe_visible_joint(keypoints, 5)
    right_shoulder = _safe_visible_joint(keypoints, 6)
    left_elbow = _safe_visible_joint(keypoints, 7)
    right_elbow = _safe_visible_joint(keypoints, 8)
    left_wrist = _safe_visible_joint(keypoints, 9)
    right_wrist = _safe_visible_joint(keypoints, 10)

    box_top = _box_metric(fallback_box, "top", default=0.0)
    box_height = _box_metric(fallback_box, "height", default=0.0)
    box_width = _box_metric(fallback_box, "width", default=0.0)
    box_center_x = _box_metric(fallback_box, "center_x", default=0.0)

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        shoulder_span = max(0.08, abs(right_shoulder[0] - left_shoulder[0]))
        chest_bottom = shoulder_y + max(0.18, box_height * 0.24)
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
            return AICameraGestureEvent.ARMS_CROSSED, _legacy_rounded_confidence(
                base=0.62,
                scale=0.18,
                attention_score=attention_score,
            )

        if (
            left_wrist is not None
            and right_wrist is not None
            and left_wrist[0] <= left_shoulder[0] - horizontal_outreach
            and right_wrist[0] >= right_shoulder[0] + horizontal_outreach
            and abs(left_wrist[1] - left_shoulder[1]) <= 0.18
            and abs(right_wrist[1] - right_shoulder[1]) <= 0.18
        ):
            return AICameraGestureEvent.TWO_HAND_DISMISS, _legacy_rounded_confidence(
                base=0.66,
                scale=0.18,
                attention_score=attention_score,
            )

        if (
            _legacy_matches_horizontal_arm_toward_center(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_center_x=shoulder_center_x,
                shoulder_y=shoulder_y,
                side="left",
            )
            and _legacy_matches_vertical_arm(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_span=shoulder_span,
            )
        ) or (
            _legacy_matches_horizontal_arm_toward_center(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_center_x=shoulder_center_x,
                shoulder_y=shoulder_y,
                side="right",
            )
            and _legacy_matches_vertical_arm(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_span=shoulder_span,
            )
        ):
            return AICameraGestureEvent.TIMEOUT_T, _legacy_rounded_confidence(
                base=0.64,
                scale=0.18,
                attention_score=attention_score,
            )

    left_command = _legacy_classify_single_arm_command(
        shoulder=left_shoulder,
        elbow=left_elbow,
        wrist=left_wrist,
        side="left",
        attention_score=attention_score,
    )
    if left_command is not None:
        return left_command

    right_command = _legacy_classify_single_arm_command(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
        attention_score=attention_score,
    )
    if right_command is not None:
        return right_command

    if _legacy_matches_wave_arm(
        shoulder=left_shoulder,
        elbow=left_elbow,
        wrist=left_wrist,
        side="left",
    ) or _legacy_matches_wave_arm(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
    ):
        return AICameraGestureEvent.WAVE, _legacy_rounded_confidence(
            base=0.60,
            scale=0.20,
            attention_score=attention_score,
        )

    wrists = [joint for joint in (left_wrist, right_wrist) if joint is not None]
    if box_width > 0.0 and box_height > 0.0:
        stop_top = box_top + max(0.10, box_height * 0.20)
        dismiss_offset = max(0.18, box_width * 0.35)
        stop_offset = max(0.14, box_width * 0.28)
        for wrist in wrists:
            wrist_x, wrist_y, _ = wrist
            center_offset = abs(wrist_x - box_center_x)
            if wrist_y <= stop_top and center_offset <= stop_offset:
                return AICameraGestureEvent.STOP, _legacy_rounded_confidence(
                    base=0.58,
                    scale=0.25,
                    attention_score=attention_score,
                )
            if wrist_y <= box_top + box_height * 0.65 and center_offset >= dismiss_offset:
                return AICameraGestureEvent.DISMISS, _legacy_rounded_confidence(
                    base=0.5,
                    scale=0.2,
                    attention_score=attention_score,
                )

    return AICameraGestureEvent.NONE, None


def classify_gesture(
    keypoints: Mapping[int, Joint],
    *,
    attention_score: float,
    fallback_box: AICameraBox,
    state: PoseTemporalState | None = None,
    timestamp_ms: int | None = None,
) -> tuple[AICameraGestureEvent, float | None]:
    """Return one conservative coarse-arm gesture classification.

    When ``state`` and ``timestamp_ms`` are supplied, the classifier behaves as a
    debounced stream classifier. Gesture events then fire on stable rising edges
    instead of every frame.
    """

    attention_score = _finite_number(attention_score, default=0.0)
    if state is None and timestamp_ms is None:
        return _classify_gesture_single_frame_legacy(
            keypoints=keypoints,
            attention_score=attention_score,
            fallback_box=fallback_box,
        )

    left_shoulder = _safe_visible_joint(keypoints, 5)
    right_shoulder = _safe_visible_joint(keypoints, 6)
    left_elbow = _safe_visible_joint(keypoints, 7)
    right_elbow = _safe_visible_joint(keypoints, 8)
    left_wrist = _safe_visible_joint(keypoints, 9)
    right_wrist = _safe_visible_joint(keypoints, 10)

    geom = _build_geometry_context(
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        fallback_box=fallback_box,
    )

    _update_wrist_history(
        state=state,
        timestamp_ms=timestamp_ms,
        side="left",
        shoulder=left_shoulder,
        wrist=left_wrist,
        geom=geom,
    )
    _update_wrist_history(
        state=state,
        timestamp_ms=timestamp_ms,
        side="right",
        shoulder=right_shoulder,
        wrist=right_wrist,
        geom=geom,
    )

    candidates: dict[AICameraGestureEvent, float] = {}

    if left_shoulder is not None and right_shoulder is not None:
        crossed_margin = max(0.03, geom.shoulder_span * 0.08)
        chest_half_height = max(0.12, geom.vertical_scale * 0.42)
        horizontal_outreach = max(0.12, geom.horizontal_scale * 0.34)

        if left_wrist is not None and right_wrist is not None:
            chest_band_score = min(
                _soft_band(
                    left_wrist[1],
                    center=geom.center_shoulder_y + chest_half_height * 0.42,
                    half_width=chest_half_height,
                    softness=chest_half_height * 0.60,
                ),
                _soft_band(
                    right_wrist[1],
                    center=geom.center_shoulder_y + chest_half_height * 0.42,
                    half_width=chest_half_height,
                    softness=chest_half_height * 0.60,
                ),
            )
            crossed_score = min(
                _soft_pass_min(left_wrist[0] - geom.center_shoulder_x, crossed_margin, softness=crossed_margin * 0.90),
                _soft_pass_min(geom.center_shoulder_x - right_wrist[0], crossed_margin, softness=crossed_margin * 0.90),
                chest_band_score,
            )
            candidates[AICameraGestureEvent.ARMS_CROSSED] = crossed_score * _joint_confidence(
                left_shoulder,
                right_shoulder,
                left_wrist,
                right_wrist,
            )

            two_hand_dismiss_score = min(
                _soft_pass_min(left_shoulder[0] - left_wrist[0], horizontal_outreach, softness=horizontal_outreach * 0.75),
                _soft_pass_min(right_wrist[0] - right_shoulder[0], horizontal_outreach, softness=horizontal_outreach * 0.75),
                _soft_pass_max(abs(left_wrist[1] - left_shoulder[1]), max(0.12, geom.vertical_scale * 0.34), softness=max(0.10, geom.vertical_scale * 0.28)),
                _soft_pass_max(abs(right_wrist[1] - right_shoulder[1]), max(0.12, geom.vertical_scale * 0.34), softness=max(0.10, geom.vertical_scale * 0.28)),
            )
            candidates[AICameraGestureEvent.TWO_HAND_DISMISS] = two_hand_dismiss_score * _joint_confidence(
                left_shoulder,
                right_shoulder,
                left_wrist,
                right_wrist,
            )

        timeout_left_score = min(
            _arm_horizontal_toward_center_score(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_center_x=geom.center_shoulder_x,
                shoulder_y=geom.center_shoulder_y,
                side="left",
                geom=geom,
            ),
            _arm_vertical_score(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                geom=geom,
            ),
        )
        timeout_right_score = min(
            _arm_horizontal_toward_center_score(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_center_x=geom.center_shoulder_x,
                shoulder_y=geom.center_shoulder_y,
                side="right",
                geom=geom,
            ),
            _arm_vertical_score(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                geom=geom,
            ),
        )
        candidates[AICameraGestureEvent.TIMEOUT_T] = max(timeout_left_score, timeout_right_score)

    for event, score in _command_scores_for_arm(
        shoulder=left_shoulder,
        elbow=left_elbow,
        wrist=left_wrist,
        side="left",
        geom=geom,
    ).items():
        candidates[event] = max(candidates.get(event, 0.0), score)

    for event, score in _command_scores_for_arm(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
        geom=geom,
    ).items():
        candidates[event] = max(candidates.get(event, 0.0), score)

    left_wave_pose = _wave_static_pose_score(
        shoulder=left_shoulder,
        elbow=left_elbow,
        wrist=left_wrist,
        side="left",
        geom=geom,
    )
    right_wave_pose = _wave_static_pose_score(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
        geom=geom,
    )
    left_wave_motion = _wave_motion_score(
        state=state,
        timestamp_ms=timestamp_ms,
        side="left",
        geom=geom,
    )
    right_wave_motion = _wave_motion_score(
        state=state,
        timestamp_ms=timestamp_ms,
        side="right",
        geom=geom,
    )

    # BREAKING: In stream mode, WAVE is treated as a temporal gesture, not a single-frame raised-hand pose.
    wave_score = max(
        min(left_wave_pose, left_wave_motion) if state is not None and timestamp_ms is not None else left_wave_pose * 0.64,
        min(right_wave_pose, right_wave_motion) if state is not None and timestamp_ms is not None else right_wave_pose * 0.64,
    )
    candidates[AICameraGestureEvent.WAVE] = max(candidates.get(AICameraGestureEvent.WAVE, 0.0), wave_score)

    wrists = [joint for joint in (left_wrist, right_wrist) if joint is not None]
    if geom.box_width > 0.0 and geom.box_height > 0.0:
        stop_top = geom.box_top + max(0.10, geom.box_height * 0.20)
        dismiss_offset = max(0.18, geom.box_width * 0.35)
        stop_offset = max(0.14, geom.box_width * 0.28)
        fallback_stop_score = 0.0
        fallback_dismiss_score = 0.0
        for wrist in wrists:
            wrist_x, wrist_y, _ = wrist
            center_offset = abs(wrist_x - geom.box_center_x)
            fallback_stop_score = max(
                fallback_stop_score,
                min(
                    _soft_pass_max(wrist_y, stop_top, softness=max(0.05, geom.box_height * 0.10)),
                    _soft_pass_max(center_offset, stop_offset, softness=max(0.05, stop_offset * 0.60)),
                )
                * _joint_confidence(wrist),
            )
            fallback_dismiss_score = max(
                fallback_dismiss_score,
                min(
                    _soft_pass_max(wrist_y, geom.box_top + geom.box_height * 0.65, softness=max(0.08, geom.box_height * 0.14)),
                    _soft_pass_min(center_offset, dismiss_offset, softness=max(0.06, dismiss_offset * 0.50)),
                )
                * _joint_confidence(wrist),
            )
        candidates[AICameraGestureEvent.STOP] = max(
            candidates.get(AICameraGestureEvent.STOP, 0.0),
            fallback_stop_score * 0.78,
        )
        candidates[AICameraGestureEvent.DISMISS] = max(
            candidates.get(AICameraGestureEvent.DISMISS, 0.0),
            fallback_dismiss_score * 0.72,
        )

    best_event, best_score = _select_best_event(candidates)
    # BREAKING: When stream state is supplied, gesture outputs become edge-triggered with cooldown.
    best_event, best_score = _debounce_gesture_event(
        event=best_event,
        score=best_score,
        state=state,
        timestamp_ms=timestamp_ms,
    )
    if best_event is AICameraGestureEvent.NONE or best_score <= 0.0:
        return AICameraGestureEvent.NONE, None
    evidence = candidates.get(best_event, best_score)
    return best_event, _rounded_confidence(
        score=best_score,
        attention_score=attention_score,
        evidence=evidence,
    )


def _body_evidence_from_centerline(
    *,
    left_shoulder: Joint | None,
    right_shoulder: Joint | None,
    left_hip: Joint | None,
    right_hip: Joint | None,
    left_knee: Joint | None,
    right_knee: Joint | None,
    left_ankle: Joint | None,
    right_ankle: Joint | None,
    world_keypoints: Mapping[int, Joint] | None,
) -> _BodyEvidence:
    """Return body evidence from metric or image-space centerline geometry."""

    world_left_shoulder = _raw_sanitized_joint(world_keypoints, 5)
    world_right_shoulder = _raw_sanitized_joint(world_keypoints, 6)
    world_left_hip = _raw_sanitized_joint(world_keypoints, 11)
    world_right_hip = _raw_sanitized_joint(world_keypoints, 12)

    shoulders_ready = left_shoulder is not None and right_shoulder is not None
    hips_ready = left_hip is not None and right_hip is not None

    source = "box"
    if world_left_shoulder is not None and world_right_shoulder is not None and world_left_hip is not None and world_right_hip is not None:
        shoulder_a = world_left_shoulder
        shoulder_b = world_right_shoulder
        hip_a = world_left_hip
        hip_b = world_right_hip
        source = "world"
    elif shoulders_ready and hips_ready:
        shoulder_a = left_shoulder
        shoulder_b = right_shoulder
        hip_a = left_hip
        hip_b = right_hip
        source = "image"
    elif left_shoulder is not None and left_hip is not None:
        shoulder_a = left_shoulder
        shoulder_b = None
        hip_a = left_hip
        hip_b = None
        source = "image-side"
    elif right_shoulder is not None and right_hip is not None:
        shoulder_a = right_shoulder
        shoulder_b = None
        hip_a = right_hip
        hip_b = None
        source = "image-side"
    else:
        return _BodyEvidence(source=source)

    if shoulder_b is not None and hip_b is not None:
        shoulder_x = (shoulder_a[0] + shoulder_b[0]) / 2.0
        shoulder_y = (shoulder_a[1] + shoulder_b[1]) / 2.0
        hip_x = (hip_a[0] + hip_b[0]) / 2.0
        hip_y = (hip_a[1] + hip_b[1]) / 2.0
        torso_conf = _joint_confidence(shoulder_a, shoulder_b, hip_a, hip_b)
    else:
        shoulder_x = shoulder_a[0]
        shoulder_y = shoulder_a[1]
        hip_x = hip_a[0]
        hip_y = hip_a[1]
        torso_conf = _joint_confidence(shoulder_a, hip_a)

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    torso_length = math.hypot(dx, dy)
    if torso_length <= 1e-6:
        return _BodyEvidence(source=source)
    torso_angle = math.degrees(abs(math.atan2(dx, dy if abs(dy) > 1e-6 else 1e-6)))

    knee_drop = None
    if left_knee is not None and right_knee is not None and hips_ready:
        knee_drop = ((left_knee[1] + right_knee[1]) / 2.0) - ((left_hip[1] + right_hip[1]) / 2.0)
        lower_conf = _joint_confidence(left_hip, right_hip, left_knee, right_knee)
    else:
        lower_conf = 0.0

    ankle_drop = None
    if left_ankle is not None and right_ankle is not None and hips_ready:
        ankle_drop = ((left_ankle[1] + right_ankle[1]) / 2.0) - ((left_hip[1] + right_hip[1]) / 2.0)
        lower_conf = max(lower_conf, _joint_confidence(left_hip, right_hip, left_ankle, right_ankle))
    elif left_ankle is not None and left_hip is not None:
        ankle_drop = left_ankle[1] - left_hip[1]
        lower_conf = max(lower_conf, _joint_confidence(left_hip, left_ankle))
    elif right_ankle is not None and right_hip is not None:
        ankle_drop = right_ankle[1] - right_hip[1]
        lower_conf = max(lower_conf, _joint_confidence(right_hip, right_ankle))

    return _BodyEvidence(
        torso_angle_from_vertical_deg=torso_angle,
        torso_length=torso_length,
        torso_confidence=torso_conf,
        knee_drop=knee_drop,
        ankle_drop=ankle_drop,
        lower_chain_confidence=lower_conf,
        source=source,
    )


def _select_best_label(
    candidates: Mapping[T, float],
    *,
    unknown_label: T,
    min_score: float,
) -> tuple[T, float]:
    """Return the best label with abstention on ambiguity."""

    if not candidates:
        return unknown_label, 0.0
    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    if best_score < min_score:
        return unknown_label, 0.0
    if len(ranked) >= 2:
        _, second_score = ranked[1]
        if second_score >= best_score - 0.05 and best_score < 0.82:
            return unknown_label, 0.0
    return best_label, _clamp01(best_score)


def _stable_body_pose(
    *,
    label: AICameraBodyPose,
    score: float,
    state: PoseTemporalState | None,
) -> AICameraBodyPose:
    """Return a smoothed body pose label."""

    if state is None:
        return label
    if label == state.body_candidate:
        state.body_candidate_frames += 1
    else:
        state.body_candidate = label
        state.body_candidate_frames = 1

    required_frames = 1 if score >= 0.86 else 2
    if state.body_candidate_frames >= required_frames:
        state.body_stable = label
    return state.body_stable


def classify_body_pose(
    keypoints: Mapping[int, Joint],
    *,
    fallback_box: AICameraBox,
    world_keypoints: Mapping[int, Joint] | None = None,
    state: PoseTemporalState | None = None,
) -> AICameraBodyPose:
    """Return one coarse, conservative body pose classification."""

    left_shoulder = _safe_visible_joint(keypoints, 5)
    right_shoulder = _safe_visible_joint(keypoints, 6)
    left_hip = _safe_visible_joint(keypoints, 11)
    right_hip = _safe_visible_joint(keypoints, 12)
    left_knee = _safe_visible_joint(keypoints, 13)
    right_knee = _safe_visible_joint(keypoints, 14)
    left_ankle = _safe_visible_joint(keypoints, 15)
    right_ankle = _safe_visible_joint(keypoints, 16)

    geom = _build_geometry_context(
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_hip=left_hip,
        right_hip=right_hip,
        left_ankle=left_ankle,
        right_ankle=right_ankle,
        fallback_box=fallback_box,
    )
    body = _body_evidence_from_centerline(
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_hip=left_hip,
        right_hip=right_hip,
        left_knee=left_knee,
        right_knee=right_knee,
        left_ankle=left_ankle,
        right_ankle=right_ankle,
        world_keypoints=world_keypoints,
    )

    box_area = _box_metric(fallback_box, "area", default=0.0)

    candidates: dict[AICameraBodyPose, float] = {
        AICameraBodyPose.UNKNOWN: 0.0,
    }

    if body.torso_angle_from_vertical_deg is not None and body.torso_length > 0.0:
        angle = body.torso_angle_from_vertical_deg
        torso_conf = max(0.20, body.torso_confidence)
        lower_conf = max(0.0, body.lower_chain_confidence)
        knee_ratio = _safe_div(_finite_number(body.knee_drop, default=-1.0), body.torso_length, default=-1.0)
        leg_ratio = _safe_div(_finite_number(body.ankle_drop, default=-1.0), body.torso_length, default=-1.0)

        lying_score = _soft_pass_min(angle, 60.0, softness=12.0) * torso_conf
        floor_score = min(
            1.0,
            lying_score * (0.92 + 0.18 * _soft_pass_min(geom.box_center_y, 0.72, softness=0.12)),
        )

        seated_score = 0.0
        # BUG-FIX: knees must actually be below hips to count as seated.
        if body.knee_drop is not None and knee_ratio >= 0.10:
            seated_score = min(
                _soft_pass_max(angle, 16.0, softness=12.0),
                _soft_pass_min(knee_ratio, 0.10, softness=0.10),
                _soft_pass_max(knee_ratio, 0.62, softness=0.20),
            )
            if body.ankle_drop is not None:
                seated_score = min(
                    seated_score,
                    _soft_pass_max(leg_ratio, 0.95, softness=0.22),
                )
            seated_score *= max(torso_conf, lower_conf)
            seated_score = min(
                1.0,
                seated_score * (0.92 + 0.08 * _soft_pass_min(geom.box_center_y, 0.56, softness=0.12)),
            )

        upright_score = 0.0
        if body.ankle_drop is not None:
            upright_score = min(
                _soft_pass_max(angle, 12.0, softness=10.0),
                _soft_pass_min(leg_ratio, 1.02, softness=0.25),
            ) * max(torso_conf, lower_conf * 0.90)

        slumped_score = min(
            _soft_pass_min(angle, 10.0, softness=8.0),
            _soft_pass_max(angle, 42.0, softness=18.0),
        ) * max(torso_conf, lower_conf * 0.85)

        # BUG-FIX: low crop position is only a weak tiebreaker, never a strong override against upright skeleton evidence.
        if geom.box_center_y > 0.68:
            slumped_score = min(1.0, slumped_score * 1.08)
        if upright_score > 0.75:
            slumped_score *= 0.60

        candidates[AICameraBodyPose.LYING_LOW] = lying_score
        candidates[AICameraBodyPose.FLOOR] = floor_score
        candidates[AICameraBodyPose.SEATED] = seated_score
        candidates[AICameraBodyPose.UPRIGHT] = upright_score
        candidates[AICameraBodyPose.SLUMPED] = slumped_score

    if box_area > 0.0:
        floor_from_box = min(
            _soft_pass_min(geom.box_width, max(0.42, geom.box_height * 1.10), softness=max(0.08, geom.box_width * 0.25)),
            _soft_pass_min(geom.box_center_y, 0.72, softness=0.14),
        ) * 0.72
        lying_from_box = min(
            _soft_pass_min(geom.box_width, max(0.42, geom.box_height * 1.05), softness=max(0.08, geom.box_width * 0.25)),
            _soft_pass_max(geom.box_height, 0.34, softness=0.12),
        ) * 0.68
        upright_from_box = min(
            _soft_pass_min(geom.box_height, 0.78, softness=0.12),
            _soft_pass_max(geom.box_width, 0.42, softness=0.10),
            _soft_pass_max(geom.box_center_y, 0.64, softness=0.12),
        ) * 0.58
        seated_from_box = min(
            _soft_pass_max(geom.box_height, 0.42, softness=0.12),
            _soft_pass_min(geom.box_height, 0.74, softness=0.12),
            _soft_pass_max(geom.box_width, 0.58, softness=0.12),
        ) * 0.52

        candidates[AICameraBodyPose.FLOOR] = max(candidates.get(AICameraBodyPose.FLOOR, 0.0), floor_from_box)
        candidates[AICameraBodyPose.LYING_LOW] = max(candidates.get(AICameraBodyPose.LYING_LOW, 0.0), lying_from_box)
        candidates[AICameraBodyPose.UPRIGHT] = max(candidates.get(AICameraBodyPose.UPRIGHT, 0.0), upright_from_box)
        candidates[AICameraBodyPose.SEATED] = max(candidates.get(AICameraBodyPose.SEATED, 0.0), seated_from_box)

    best_label, best_score = _select_best_label(
        candidates,
        unknown_label=AICameraBodyPose.UNKNOWN,
        min_score=0.50,
    )
    return _stable_body_pose(label=best_label, score=best_score, state=state)


__all__ = [
    "PoseTemporalState",
    "classify_body_pose",
    "classify_gesture",
    "matches_horizontal_arm_toward_center",
    "matches_vertical_arm",
    "matches_wave_arm",
]
