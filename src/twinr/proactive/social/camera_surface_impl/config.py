"""Configuration for the proactive camera stabilization surface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..gesture_calibration import GestureCalibrationProfile
from .coercion import coerce_optional_ratio
from .validation import (
    coerce_non_negative_float,
    coerce_positive_float,
    coerce_positive_int,
    require_bounded_float,
    require_bounded_ratio,
    require_non_negative_float,
    require_positive_int,
)

MIN_CENTER_SMOOTHING_ALPHA = 0.1
MAX_CENTER_SMOOTHING_ALPHA = 1.0
MIN_CENTER_DEADBAND = 0.0
MAX_CENTER_DEADBAND = 0.25
MIN_CENTER_SMOOTHING_WINDOW_S = 0.1
MAX_CENTER_SMOOTHING_WINDOW_S = 10.0


@dataclass(frozen=True, slots=True)
class ProactiveCameraSurfaceConfig:
    """Store stabilization rules for camera-derived automation signals."""

    person_visible_on_samples: int = 1
    person_visible_off_samples: int = 2
    person_visible_unknown_hold_s: float = 9.0
    person_visible_event_cooldown_s: float = 9.0
    person_recently_visible_window_s: float = 30.0
    person_returned_absence_s: float = 20.0 * 60.0
    looking_toward_device_on_samples: int = 1
    looking_toward_device_off_samples: int = 2
    looking_toward_device_unknown_hold_s: float = 9.0
    person_near_device_on_samples: int = 1
    person_near_device_off_samples: int = 2
    person_near_device_unknown_hold_s: float = 9.0
    engaged_with_device_on_samples: int = 1
    engaged_with_device_off_samples: int = 2
    engaged_with_device_unknown_hold_s: float = 9.0
    showing_intent_on_samples: int = 1
    showing_intent_off_samples: int = 2
    showing_intent_unknown_hold_s: float = 9.0
    showing_intent_event_cooldown_s: float = 9.0
    hand_or_object_near_camera_on_samples: int = 1
    hand_or_object_near_camera_off_samples: int = 2
    hand_or_object_near_camera_unknown_hold_s: float = 9.0
    hand_or_object_near_camera_event_cooldown_s: float = 9.0
    motion_event_cooldown_s: float = 9.0
    gesture_event_cooldown_s: float = 9.0
    fine_hand_explicit_hold_s: float = 0.45
    fine_hand_explicit_confirm_samples: int = 1
    fine_hand_explicit_min_confidence: float = 0.72
    gesture_calibration: GestureCalibrationProfile = field(default_factory=GestureCalibrationProfile.defaults)
    primary_person_center_smoothing_alpha: float = 0.58
    primary_person_center_deadband: float = 0.028
    primary_person_center_smoothing_window_s: float = 1.4
    object_on_samples: int = 2
    object_off_samples: int = 2
    object_unknown_hold_s: float = 9.0
    secondary_unknown_hold_s: float = 9.0

    def __post_init__(self) -> None:
        """Reject malformed debounce and cooldown configuration eagerly."""

        require_positive_int(self.person_visible_on_samples, field_name="person_visible_on_samples")
        require_positive_int(self.person_visible_off_samples, field_name="person_visible_off_samples")
        require_non_negative_float(self.person_visible_unknown_hold_s, field_name="person_visible_unknown_hold_s")
        require_non_negative_float(self.person_visible_event_cooldown_s, field_name="person_visible_event_cooldown_s")
        require_non_negative_float(self.person_recently_visible_window_s, field_name="person_recently_visible_window_s")
        require_non_negative_float(self.person_returned_absence_s, field_name="person_returned_absence_s")
        require_positive_int(self.looking_toward_device_on_samples, field_name="looking_toward_device_on_samples")
        require_positive_int(self.looking_toward_device_off_samples, field_name="looking_toward_device_off_samples")
        require_non_negative_float(
            self.looking_toward_device_unknown_hold_s,
            field_name="looking_toward_device_unknown_hold_s",
        )
        require_positive_int(self.person_near_device_on_samples, field_name="person_near_device_on_samples")
        require_positive_int(self.person_near_device_off_samples, field_name="person_near_device_off_samples")
        require_non_negative_float(
            self.person_near_device_unknown_hold_s,
            field_name="person_near_device_unknown_hold_s",
        )
        require_positive_int(self.engaged_with_device_on_samples, field_name="engaged_with_device_on_samples")
        require_positive_int(self.engaged_with_device_off_samples, field_name="engaged_with_device_off_samples")
        require_non_negative_float(
            self.engaged_with_device_unknown_hold_s,
            field_name="engaged_with_device_unknown_hold_s",
        )
        require_positive_int(self.showing_intent_on_samples, field_name="showing_intent_on_samples")
        require_positive_int(self.showing_intent_off_samples, field_name="showing_intent_off_samples")
        require_non_negative_float(self.showing_intent_unknown_hold_s, field_name="showing_intent_unknown_hold_s")
        require_non_negative_float(self.showing_intent_event_cooldown_s, field_name="showing_intent_event_cooldown_s")
        require_positive_int(
            self.hand_or_object_near_camera_on_samples,
            field_name="hand_or_object_near_camera_on_samples",
        )
        require_positive_int(
            self.hand_or_object_near_camera_off_samples,
            field_name="hand_or_object_near_camera_off_samples",
        )
        require_non_negative_float(
            self.hand_or_object_near_camera_unknown_hold_s,
            field_name="hand_or_object_near_camera_unknown_hold_s",
        )
        require_non_negative_float(
            self.hand_or_object_near_camera_event_cooldown_s,
            field_name="hand_or_object_near_camera_event_cooldown_s",
        )
        require_non_negative_float(self.motion_event_cooldown_s, field_name="motion_event_cooldown_s")
        require_non_negative_float(self.gesture_event_cooldown_s, field_name="gesture_event_cooldown_s")
        require_non_negative_float(self.fine_hand_explicit_hold_s, field_name="fine_hand_explicit_hold_s")
        require_positive_int(
            self.fine_hand_explicit_confirm_samples,
            field_name="fine_hand_explicit_confirm_samples",
        )
        require_bounded_ratio(
            self.fine_hand_explicit_min_confidence,
            field_name="fine_hand_explicit_min_confidence",
            minimum=0.0,
            maximum=1.0,
        )
        require_bounded_ratio(
            self.primary_person_center_smoothing_alpha,
            field_name="primary_person_center_smoothing_alpha",
            minimum=MIN_CENTER_SMOOTHING_ALPHA,
            maximum=MAX_CENTER_SMOOTHING_ALPHA,
        )
        require_bounded_ratio(
            self.primary_person_center_deadband,
            field_name="primary_person_center_deadband",
            minimum=MIN_CENTER_DEADBAND,
            maximum=MAX_CENTER_DEADBAND,
        )
        require_bounded_float(
            self.primary_person_center_smoothing_window_s,
            field_name="primary_person_center_smoothing_window_s",
            minimum=MIN_CENTER_SMOOTHING_WINDOW_S,
            maximum=MAX_CENTER_SMOOTHING_WINDOW_S,
        )
        require_positive_int(self.object_on_samples, field_name="object_on_samples")
        require_positive_int(self.object_off_samples, field_name="object_off_samples")
        require_non_negative_float(self.object_unknown_hold_s, field_name="object_unknown_hold_s")
        require_non_negative_float(self.secondary_unknown_hold_s, field_name="secondary_unknown_hold_s")

    @classmethod
    def from_config(cls, config: object) -> "ProactiveCameraSurfaceConfig":
        """Build one cadence-aware camera surface config from Twinr config."""

        interval_s = coerce_positive_float(getattr(config, "proactive_capture_interval_s", 6.0), default=6.0)
        attention_refresh_s = coerce_positive_float(
            getattr(config, "display_attention_refresh_interval_s", interval_s),
            default=interval_s,
        )
        unknown_hold_s = max(interval_s + 1.0, interval_s * 1.5)
        cooldown_s = max(interval_s, interval_s * 1.5)
        gesture_cooldown_s = max(0.8, min(2.0, attention_refresh_s * 2.0))
        fine_hand_explicit_hold_s = max(0.2, min(0.8, attention_refresh_s * 0.75))
        fine_hand_explicit_min_confidence = coerce_optional_ratio(
            getattr(config, "proactive_local_camera_fine_hand_explicit_min_confidence", 0.72)
        )
        # HDMI face-follow must still suppress tiny box wobble, but the old
        # 4x refresh window made person anchors feel visibly delayed on the Pi.
        center_smoothing_window_s = max(
            MIN_CENTER_SMOOTHING_WINDOW_S,
            min(MAX_CENTER_SMOOTHING_WINDOW_S, attention_refresh_s * 1.75),
        )
        center_deadband = max(
            0.012,
            min(MAX_CENTER_DEADBAND, attention_refresh_s * 0.05),
        )
        return cls(
            person_visible_unknown_hold_s=unknown_hold_s,
            person_visible_event_cooldown_s=cooldown_s,
            person_recently_visible_window_s=max(30.0, interval_s * 5.0),
            person_returned_absence_s=coerce_positive_float(
                getattr(config, "proactive_person_returned_absence_s", 20.0 * 60.0),
                default=20.0 * 60.0,
            ),
            looking_toward_device_unknown_hold_s=unknown_hold_s,
            person_near_device_unknown_hold_s=unknown_hold_s,
            engaged_with_device_unknown_hold_s=unknown_hold_s,
            showing_intent_unknown_hold_s=unknown_hold_s,
            showing_intent_event_cooldown_s=cooldown_s,
            hand_or_object_near_camera_unknown_hold_s=unknown_hold_s,
            hand_or_object_near_camera_event_cooldown_s=cooldown_s,
            motion_event_cooldown_s=cooldown_s,
            gesture_event_cooldown_s=gesture_cooldown_s,
            fine_hand_explicit_hold_s=coerce_non_negative_float(
                getattr(config, "proactive_local_camera_fine_hand_explicit_hold_s", fine_hand_explicit_hold_s),
                default=fine_hand_explicit_hold_s,
            ),
            fine_hand_explicit_confirm_samples=coerce_positive_int(
                getattr(config, "proactive_local_camera_fine_hand_explicit_confirm_samples", 1),
                default=1,
            ),
            fine_hand_explicit_min_confidence=(
                0.72 if fine_hand_explicit_min_confidence is None else float(fine_hand_explicit_min_confidence)
            ),
            gesture_calibration=GestureCalibrationProfile.from_runtime_config(config),
            primary_person_center_smoothing_alpha=0.76,
            primary_person_center_deadband=center_deadband,
            primary_person_center_smoothing_window_s=center_smoothing_window_s,
            object_unknown_hold_s=unknown_hold_s,
            secondary_unknown_hold_s=unknown_hold_s,
        )
