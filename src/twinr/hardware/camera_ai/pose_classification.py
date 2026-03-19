"""Classify bounded coarse body and single-frame arm states from keypoints."""

from __future__ import annotations

import math

from .config import _clamp_ratio
from .models import AICameraBodyPose, AICameraBox, AICameraGestureEvent
from .pose_features import visible_joint


def _finite_number(value: object, *, default: float = 0.0) -> float:
    """Return a finite float or a safe default."""

    # AUDIT-FIX(#3): Non-finite model outputs must fail closed so geometry and confidence math stay stable.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sanitize_joint(
    joint: tuple[float, float, float] | None,
) -> tuple[float, float, float] | None:
    """Return a finite 3-value joint tuple or None."""

    # AUDIT-FIX(#3): Reject malformed or non-finite joints instead of letting NaN/inf propagate through rules.
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
    return (x, y, confidence)


def _safe_visible_joint(
    keypoints: dict[int, tuple[float, float, float]],
    index: int,
) -> tuple[float, float, float] | None:
    """Return one sanitized visible joint."""

    return _sanitize_joint(visible_joint(keypoints, index))


def _box_metric(box: AICameraBox, attr: str, *, default: float = 0.0) -> float:
    """Return one finite fallback-box metric."""

    # AUDIT-FIX(#3): Fallback-box metrics also need finite coercion because camera pipelines can emit invalid floats.
    return _finite_number(getattr(box, attr, default), default=default)


def _rounded_confidence(*, base: float, scale: float, attention_score: float) -> float:
    """Return one safe rounded confidence."""

    bounded = _finite_number(
        _clamp_ratio(base + scale * attention_score, default=base),
        default=base,
    )
    return round(bounded, 3)


def _classify_single_arm_command(
    *,
    shoulder: tuple[float, float, float] | None,
    elbow: tuple[float, float, float] | None,
    wrist: tuple[float, float, float] | None,
    side: str,
    attention_score: float,
) -> tuple[AICameraGestureEvent, float] | None:
    """Return one explicit single-arm command gesture if present."""

    if shoulder is None or wrist is None:
        return None
    if side not in {"left", "right"}:
        # AUDIT-FIX(#5): Unknown side values must fail closed instead of being interpreted as the opposite arm.
        return None
    if wrist[1] < shoulder[1] - 0.08 and abs(wrist[0] - shoulder[0]) <= 0.16:
        return AICameraGestureEvent.STOP, _rounded_confidence(
            base=0.7,
            scale=0.3,
            attention_score=attention_score,
        )
    outward = wrist[0] < shoulder[0] - 0.18 if side == "left" else wrist[0] > shoulder[0] + 0.18
    if outward and abs(wrist[1] - shoulder[1]) <= 0.16:
        return AICameraGestureEvent.DISMISS, _rounded_confidence(
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
        return AICameraGestureEvent.CONFIRM, _rounded_confidence(
            base=0.55,
            scale=0.2,
            attention_score=attention_score,
        )
    return None


def matches_wave_arm(
    *,
    shoulder: tuple[float, float, float] | None,
    elbow: tuple[float, float, float] | None,
    wrist: tuple[float, float, float] | None,
    side: str,
) -> bool:
    """Return whether one arm geometry looks like a simple raised wave pose."""

    if shoulder is None or elbow is None or wrist is None:
        return False
    if side not in {"left", "right"}:
        # AUDIT-FIX(#5): Unknown side labels must not silently behave like "right".
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


def matches_vertical_arm(
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
        and wrist[1] < elbow[1]
        and wrist[1] <= shoulder[1] - 0.06  # AUDIT-FIX(#2): Require the wrist to be genuinely raised above shoulder level.
        and elbow[1] <= shoulder[1] + 0.10  # AUDIT-FIX(#2): A lowered vertical arm must not satisfy TIMEOUT_T.
    )


def matches_horizontal_arm_toward_center(
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
    if side not in {"left", "right"}:
        # AUDIT-FIX(#5): Unknown side labels must fail closed for exported helper APIs.
        return False
    toward_center = wrist[0] > shoulder[0] if side == "left" else wrist[0] < shoulder[0]
    return (
        toward_center
        and abs(wrist[1] - elbow[1]) <= 0.08
        and abs(elbow[1] - shoulder_y) <= 0.16
        and abs(wrist[0] - shoulder_center_x) <= 0.16
    )


def classify_gesture(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    attention_score: float,
    fallback_box: AICameraBox,
) -> tuple[AICameraGestureEvent, float | None]:
    """Return one conservative coarse-arm gesture classification."""

    attention_score = _finite_number(attention_score, default=0.0)  # AUDIT-FIX(#3): Guard confidence math from NaN/inf scores.
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
            return AICameraGestureEvent.ARMS_CROSSED, _rounded_confidence(
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
            return AICameraGestureEvent.TWO_HAND_DISMISS, _rounded_confidence(
                base=0.66,
                scale=0.18,
                attention_score=attention_score,
            )

        if (
            matches_horizontal_arm_toward_center(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_center_x=shoulder_center_x,
                shoulder_y=shoulder_y,
                side="left",
            )
            and matches_vertical_arm(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_span=shoulder_span,
            )
        ) or (
            matches_horizontal_arm_toward_center(
                shoulder=right_shoulder,
                elbow=right_elbow,
                wrist=right_wrist,
                shoulder_center_x=shoulder_center_x,
                shoulder_y=shoulder_y,
                side="right",
            )
            and matches_vertical_arm(
                shoulder=left_shoulder,
                elbow=left_elbow,
                wrist=left_wrist,
                shoulder_span=shoulder_span,
            )
        ):
            return AICameraGestureEvent.TIMEOUT_T, _rounded_confidence(
                base=0.64,
                scale=0.18,
                attention_score=attention_score,
            )

    left_command = _classify_single_arm_command(
        shoulder=left_shoulder,
        elbow=left_elbow,
        wrist=left_wrist,
        side="left",
        attention_score=attention_score,
    )
    if left_command is not None:
        # AUDIT-FIX(#4): Explicit command gestures must outrank wave to avoid dropping senior intent.
        return left_command

    right_command = _classify_single_arm_command(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
        attention_score=attention_score,
    )
    if right_command is not None:
        # AUDIT-FIX(#4): Explicit command gestures must outrank wave to avoid dropping senior intent.
        return right_command

    if matches_wave_arm(shoulder=left_shoulder, elbow=left_elbow, wrist=left_wrist, side="left") or matches_wave_arm(
        shoulder=right_shoulder,
        elbow=right_elbow,
        wrist=right_wrist,
        side="right",
    ):
        return AICameraGestureEvent.WAVE, _rounded_confidence(
            base=0.60,
            scale=0.20,
            attention_score=attention_score,
        )

    # AUDIT-FIX(#6): Reuse cached wrist joints so one classification pass cannot observe inconsistent wrist states.
    wrists = [joint for joint in (left_wrist, right_wrist) if joint is not None]
    if box_width > 0.0 and box_height > 0.0:
        stop_top = box_top + max(0.10, box_height * 0.20)
        dismiss_offset = max(0.18, box_width * 0.35)
        stop_offset = max(0.14, box_width * 0.28)
        for wrist in wrists:
            wrist_x, wrist_y, _ = wrist
            center_offset = abs(wrist_x - box_center_x)
            if wrist_y <= stop_top and center_offset <= stop_offset:
                return AICameraGestureEvent.STOP, _rounded_confidence(
                    base=0.58,
                    scale=0.25,
                    attention_score=attention_score,
                )
            if wrist_y <= box_top + box_height * 0.65 and center_offset >= dismiss_offset:
                return AICameraGestureEvent.DISMISS, _rounded_confidence(
                    base=0.5,
                    scale=0.2,
                    attention_score=attention_score,
                )

    return AICameraGestureEvent.NONE, None


def classify_body_pose(
    keypoints: dict[int, tuple[float, float, float]],
    *,
    fallback_box: AICameraBox,
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

    box_area = _box_metric(fallback_box, "area", default=0.0)
    box_width = _box_metric(fallback_box, "width", default=0.0)
    box_height = _box_metric(fallback_box, "height", default=0.0)
    box_center_y = _box_metric(fallback_box, "center_y", default=0.0)

    shoulders_ready = left_shoulder is not None and right_shoulder is not None
    hips_ready = left_hip is not None and right_hip is not None

    if shoulders_ready and hips_ready:
        # AUDIT-FIX(#1): Treat the fallback box as secondary evidence; valid torso keypoints must win over coarse crop geometry.
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2.0
        hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
        torso_vertical = max(0.0, hip_center_y - shoulder_center_y)
        torso_horizontal = max(0.02, abs(hip_center_x - shoulder_center_x))
        torso_ratio = torso_vertical / torso_horizontal
        if torso_ratio <= 0.65:
            return AICameraBodyPose.LYING_LOW if box_center_y < 0.72 else AICameraBodyPose.FLOOR
        if left_knee is not None and right_knee is not None and min(left_knee[2], right_knee[2]) >= 0.20:
            knee_center_y = (left_knee[1] + right_knee[1]) / 2.0
            if knee_center_y - hip_center_y <= 0.18:
                return AICameraBodyPose.SEATED
        if torso_ratio < 1.1 or box_center_y > 0.68:
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
        torso_horizontal = max(0.03, abs(side_hip[0] - side_shoulder[0]), box_width * 0.12)
        torso_ratio = torso_vertical / torso_horizontal
        if torso_ratio <= 0.65:
            return AICameraBodyPose.LYING_LOW if box_center_y < 0.72 else AICameraBodyPose.FLOOR
        if side_knee is not None and (side_knee[1] - side_hip[1]) <= 0.16:
            return AICameraBodyPose.SEATED
        if side_ankle is not None and (side_ankle[1] - side_hip[1]) >= 0.18:
            if torso_ratio < 0.95 or box_center_y > 0.68:
                return AICameraBodyPose.SLUMPED
            return AICameraBodyPose.UPRIGHT
        if torso_ratio < 0.95:
            return AICameraBodyPose.SLUMPED
        if box_height >= 0.68 and box_width <= 0.46:
            return AICameraBodyPose.UPRIGHT
    if side_hip is not None and side_knee is not None:
        if (side_knee[1] - side_hip[1]) <= 0.15:
            return AICameraBodyPose.SEATED
        if side_ankle is not None and (side_ankle[1] - side_hip[1]) >= 0.20 and box_height >= 0.70:
            return AICameraBodyPose.UPRIGHT

    if box_area <= 0.0:
        # AUDIT-FIX(#1): An invalid fallback box is no longer allowed to suppress otherwise usable keypoint evidence.
        return AICameraBodyPose.UNKNOWN
    if box_width >= 0.42 and box_height <= 0.28 and box_center_y >= 0.72:
        return AICameraBodyPose.FLOOR
    if box_width >= max(0.42, box_height * 1.15):
        return AICameraBodyPose.LYING_LOW
    if box_height >= 0.78 and box_width <= 0.42 and box_center_y <= 0.62:
        return AICameraBodyPose.UPRIGHT
    if 0.42 <= box_height <= 0.72 and box_width <= 0.56 and box_center_y >= 0.56:
        return AICameraBodyPose.SEATED
    return AICameraBodyPose.UNKNOWN


__all__ = [
    "classify_body_pose",
    "classify_gesture",
    "matches_horizontal_arm_toward_center",
    "matches_vertical_arm",
    "matches_wave_arm",
]