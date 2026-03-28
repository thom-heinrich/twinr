"""Normalize Twinr config into one bounded attention-servo config object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinr.agent.base_agent.config import TwinrConfig

from .constants import (
    _DEFAULT_CENTER_PULSE_WIDTH_US,
    _DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S,
    _DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US,
    _DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US,
    _DEFAULT_CONTINUOUS_RETURN_TO_ZERO_AFTER_S,
    _DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES,
    _DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES,
    _DEFAULT_CONTROL_MODE,
    _DEFAULT_DEADBAND,
    _DEFAULT_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES,
    _DEFAULT_ESTIMATED_ZERO_MOVE_DUTY_CYCLE,
    _DEFAULT_ESTIMATED_ZERO_MOVE_PERIOD_S,
    _DEFAULT_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US,
    _DEFAULT_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES,
    _DEFAULT_ESTIMATED_ZERO_SPEED_SCALE,
    _DEFAULT_EXIT_ACTIVATION_DELAY_S,
    _DEFAULT_EXIT_COOLDOWN_S,
    _DEFAULT_EXIT_FOLLOW_MAX_DEGREES,
    _DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE,
    _DEFAULT_EXIT_SETTLE_HOLD_S,
    _DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD,
    _DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD,
    _DEFAULT_FOLLOW_EXIT_ONLY,
    _DEFAULT_IDLE_RELEASE_S,
    _DEFAULT_LOSS_EXTRAPOLATION_GAIN,
    _DEFAULT_LOSS_EXTRAPOLATION_S,
    _DEFAULT_MAX_ACCELERATION_US_PER_S2,
    _DEFAULT_MAX_JERK_US_PER_S3,
    _DEFAULT_MAX_PULSE_WIDTH_US,
    _DEFAULT_MAX_STEP_US,
    _DEFAULT_MAX_VELOCITY_US_PER_S,
    _DEFAULT_MECHANICAL_RANGE_DEGREES,
    _DEFAULT_MIN_COMMAND_DELTA_US,
    _DEFAULT_MIN_CONFIDENCE,
    _DEFAULT_MIN_PULSE_WIDTH_US,
    _DEFAULT_REFERENCE_INTERVAL_S,
    _DEFAULT_REST_MAX_ACCELERATION_US_PER_S2,
    _DEFAULT_REST_MAX_JERK_US_PER_S3,
    _DEFAULT_REST_MAX_VELOCITY_US_PER_S,
    _DEFAULT_SERVO_DRIVER,
    _DEFAULT_SETTLED_RELEASE_S,
    _DEFAULT_SOFT_LIMIT_MARGIN_US,
    _DEFAULT_STATE_PATH,
    _DEFAULT_TARGET_HOLD_S,
    _DEFAULT_TARGET_SMOOTHING_S,
    _DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE,
    _DEFAULT_VISIBLE_RECENTER_INTERVAL_S,
    _DEFAULT_VISIBLE_RETARGET_TOLERANCE_US,
)

from .common import _bounded_float, _bounded_int

@dataclass(frozen=True, slots=True)
class AttentionServoConfig:
    """Store bounded attention-servo tuning from the global Twinr config."""

    enabled: bool = False
    driver: str = _DEFAULT_SERVO_DRIVER
    control_mode: str = _DEFAULT_CONTROL_MODE
    maestro_device: str | None = None
    peer_base_url: str | None = None
    peer_timeout_s: float = 1.5
    state_path: str = _DEFAULT_STATE_PATH
    estimated_zero_max_uncertainty_degrees: float = _DEFAULT_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES
    estimated_zero_settle_tolerance_degrees: float = _DEFAULT_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES
    estimated_zero_speed_scale: float = _DEFAULT_ESTIMATED_ZERO_SPEED_SCALE
    estimated_zero_move_pulse_delta_us: int = _DEFAULT_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US
    estimated_zero_move_period_s: float = _DEFAULT_ESTIMATED_ZERO_MOVE_PERIOD_S
    estimated_zero_move_duty_cycle: float = _DEFAULT_ESTIMATED_ZERO_MOVE_DUTY_CYCLE
    continuous_return_to_zero_after_s: float = _DEFAULT_CONTINUOUS_RETURN_TO_ZERO_AFTER_S
    gpio_chip: str = "gpiochip0"
    gpio: int | None = None
    invert_direction: bool = False
    target_hold_s: float = _DEFAULT_TARGET_HOLD_S
    loss_extrapolation_s: float = _DEFAULT_LOSS_EXTRAPOLATION_S
    loss_extrapolation_gain: float = _DEFAULT_LOSS_EXTRAPOLATION_GAIN
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    hold_min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    deadband: float = _DEFAULT_DEADBAND
    min_pulse_width_us: int = _DEFAULT_MIN_PULSE_WIDTH_US
    center_pulse_width_us: int = _DEFAULT_CENTER_PULSE_WIDTH_US
    max_pulse_width_us: int = _DEFAULT_MAX_PULSE_WIDTH_US
    max_step_us: int = _DEFAULT_MAX_STEP_US
    target_smoothing_s: float = _DEFAULT_TARGET_SMOOTHING_S
    max_velocity_us_per_s: float = _DEFAULT_MAX_VELOCITY_US_PER_S
    max_acceleration_us_per_s2: float = _DEFAULT_MAX_ACCELERATION_US_PER_S2
    max_jerk_us_per_s3: float = _DEFAULT_MAX_JERK_US_PER_S3
    rest_max_velocity_us_per_s: float = _DEFAULT_REST_MAX_VELOCITY_US_PER_S
    rest_max_acceleration_us_per_s2: float = _DEFAULT_REST_MAX_ACCELERATION_US_PER_S2
    rest_max_jerk_us_per_s3: float = _DEFAULT_REST_MAX_JERK_US_PER_S3
    min_command_delta_us: int = _DEFAULT_MIN_COMMAND_DELTA_US
    visible_retarget_tolerance_us: int = _DEFAULT_VISIBLE_RETARGET_TOLERANCE_US
    reference_interval_s: float = _DEFAULT_REFERENCE_INTERVAL_S
    soft_limit_margin_us: int = _DEFAULT_SOFT_LIMIT_MARGIN_US
    idle_release_s: float = _DEFAULT_IDLE_RELEASE_S
    settled_release_s: float = _DEFAULT_SETTLED_RELEASE_S
    follow_exit_only: bool = _DEFAULT_FOLLOW_EXIT_ONLY
    visible_recenter_interval_s: float = _DEFAULT_VISIBLE_RECENTER_INTERVAL_S
    visible_recenter_center_tolerance: float = _DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE
    mechanical_range_degrees: float = _DEFAULT_MECHANICAL_RANGE_DEGREES
    exit_follow_max_degrees: float = _DEFAULT_EXIT_FOLLOW_MAX_DEGREES
    exit_activation_delay_s: float = _DEFAULT_EXIT_ACTIVATION_DELAY_S
    exit_settle_hold_s: float = _DEFAULT_EXIT_SETTLE_HOLD_S
    exit_reacquire_center_tolerance: float = _DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE
    exit_visible_edge_threshold: float = _DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD
    exit_visible_box_edge_threshold: float = _DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD
    exit_cooldown_s: float = _DEFAULT_EXIT_COOLDOWN_S
    continuous_max_speed_degrees_per_s: float = _DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S
    continuous_slow_zone_degrees: float = _DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES
    continuous_stop_tolerance_degrees: float = _DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES
    continuous_min_speed_pulse_delta_us: int = _DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US
    continuous_max_speed_pulse_delta_us: int = _DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US

    @property
    def uses_continuous_rotation(self) -> bool:
        return str(self.control_mode or _DEFAULT_CONTROL_MODE).strip().lower() == "continuous_rotation"

    @property
    def safe_min_pulse_width_us(self) -> int:
        """Return the calibrated lower pulse bound after the soft margin."""

        return min(
            self.center_pulse_width_us,
            self.min_pulse_width_us + self._bounded_left_margin_us(),
        )

    @property
    def safe_max_pulse_width_us(self) -> int:
        """Return the calibrated upper pulse bound after the soft margin."""

        return max(
            self.center_pulse_width_us,
            self.max_pulse_width_us - self._bounded_right_margin_us(),
        )

    def _bounded_left_margin_us(self) -> int:
        return min(self.soft_limit_margin_us, max(0, self.center_pulse_width_us - self.min_pulse_width_us))

    def _bounded_right_margin_us(self) -> int:
        return min(self.soft_limit_margin_us, max(0, self.max_pulse_width_us - self.center_pulse_width_us))

    @property
    def exit_follow_offset_limit(self) -> float:
        """Return the maximum normalized left/right offset for exit-follow."""

        half_range_degrees = max(1.0, self.mechanical_range_degrees * 0.5)
        return max(0.0, min(1.0, self.exit_follow_max_degrees / half_range_degrees))

    @classmethod
    def from_config(cls, config: TwinrConfig) -> "AttentionServoConfig":
        """Build one bounded servo config from the global Twinr config."""

        driver = str(getattr(config, "attention_servo_driver", _DEFAULT_SERVO_DRIVER) or _DEFAULT_SERVO_DRIVER)
        control_mode = str(getattr(config, "attention_servo_control_mode", _DEFAULT_CONTROL_MODE) or _DEFAULT_CONTROL_MODE)
        maestro_device = getattr(config, "attention_servo_maestro_device", None)
        peer_base_url = getattr(config, "attention_servo_peer_base_url", None)
        peer_timeout_s = getattr(config, "attention_servo_peer_timeout_s", 1.5)
        raw_state_path = getattr(config, "attention_servo_state_path", _DEFAULT_STATE_PATH)
        resolved_state_path = Path(str(raw_state_path or _DEFAULT_STATE_PATH)).expanduser()
        if not resolved_state_path.is_absolute():
            project_root = Path(str(getattr(config, "project_root", ".") or ".")).expanduser().resolve(strict=False)
            resolved_state_path = (project_root / resolved_state_path).resolve(strict=False)
        configured_gpio = getattr(config, "attention_servo_gpio", None)
        if str(driver).strip().lower() in {"pololu_maestro", "peer_pololu_maestro"}:
            configured_gpio = getattr(config, "attention_servo_maestro_channel", None)
        min_pulse = _bounded_int(
            getattr(config, "attention_servo_min_pulse_width_us", _DEFAULT_MIN_PULSE_WIDTH_US),
            default=_DEFAULT_MIN_PULSE_WIDTH_US,
            minimum=500,
            maximum=2500,
        )
        max_pulse = _bounded_int(
            getattr(config, "attention_servo_max_pulse_width_us", _DEFAULT_MAX_PULSE_WIDTH_US),
            default=_DEFAULT_MAX_PULSE_WIDTH_US,
            minimum=500,
            maximum=2500,
        )
        if max_pulse < min_pulse:
            min_pulse, max_pulse = max_pulse, min_pulse
        center_pulse = _bounded_int(
            getattr(config, "attention_servo_center_pulse_width_us", _DEFAULT_CENTER_PULSE_WIDTH_US),
            default=_DEFAULT_CENTER_PULSE_WIDTH_US,
            minimum=min_pulse,
            maximum=max_pulse,
        )
        mechanical_range_degrees = _bounded_float(
            getattr(
                config,
                "attention_servo_mechanical_range_degrees",
                _DEFAULT_MECHANICAL_RANGE_DEGREES,
            ),
            default=_DEFAULT_MECHANICAL_RANGE_DEGREES,
            minimum=30.0,
            maximum=360.0,
        )
        target_hold_s = _bounded_float(
            getattr(config, "attention_servo_target_hold_s", _DEFAULT_TARGET_HOLD_S),
            default=_DEFAULT_TARGET_HOLD_S,
            minimum=0.0,
            maximum=5.0,
        )
        return cls(
            enabled=bool(getattr(config, "attention_servo_enabled", False)),
            driver=driver,
            control_mode=control_mode,
            maestro_device=None if maestro_device is None else (str(maestro_device).strip() or None),
            peer_base_url=None if peer_base_url is None else (str(peer_base_url).strip().rstrip("/") or None),
            peer_timeout_s=_bounded_float(
                peer_timeout_s,
                default=1.5,
                minimum=0.1,
                maximum=30.0,
            ),
            state_path=str(resolved_state_path),
            estimated_zero_max_uncertainty_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_estimated_zero_max_uncertainty_degrees",
                    _DEFAULT_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES,
                ),
                default=_DEFAULT_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES,
                minimum=0.0,
                maximum=180.0,
            ),
            estimated_zero_settle_tolerance_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_estimated_zero_settle_tolerance_degrees",
                    _DEFAULT_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES,
                ),
                default=_DEFAULT_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES,
                minimum=0.0,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            estimated_zero_speed_scale=_bounded_float(
                getattr(
                    config,
                    "attention_servo_estimated_zero_speed_scale",
                    _DEFAULT_ESTIMATED_ZERO_SPEED_SCALE,
                ),
                default=_DEFAULT_ESTIMATED_ZERO_SPEED_SCALE,
                minimum=0.0,
                maximum=1.0,
            ),
            estimated_zero_move_pulse_delta_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_estimated_zero_move_pulse_delta_us",
                    _DEFAULT_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US,
                ),
                default=_DEFAULT_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US,
                minimum=0,
                maximum=max(1, min(center_pulse - min_pulse, max_pulse - center_pulse)),
            ),
            estimated_zero_move_period_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_estimated_zero_move_period_s",
                    _DEFAULT_ESTIMATED_ZERO_MOVE_PERIOD_S,
                ),
                default=_DEFAULT_ESTIMATED_ZERO_MOVE_PERIOD_S,
                minimum=0.05,
                maximum=10.0,
            ),
            estimated_zero_move_duty_cycle=_bounded_float(
                getattr(
                    config,
                    "attention_servo_estimated_zero_move_duty_cycle",
                    _DEFAULT_ESTIMATED_ZERO_MOVE_DUTY_CYCLE,
                ),
                default=_DEFAULT_ESTIMATED_ZERO_MOVE_DUTY_CYCLE,
                minimum=0.05,
                maximum=1.0,
            ),
            continuous_return_to_zero_after_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_return_to_zero_after_s",
                    _DEFAULT_CONTINUOUS_RETURN_TO_ZERO_AFTER_S,
                ),
                default=_DEFAULT_CONTINUOUS_RETURN_TO_ZERO_AFTER_S,
                minimum=0.0,
                maximum=3600.0,
            ),
            gpio_chip=str(getattr(config, "gpio_chip", "gpiochip0") or "gpiochip0"),
            gpio=configured_gpio,
            invert_direction=bool(getattr(config, "attention_servo_invert_direction", False)),
            target_hold_s=target_hold_s,
            loss_extrapolation_s=_bounded_float(
                getattr(config, "attention_servo_loss_extrapolation_s", _DEFAULT_LOSS_EXTRAPOLATION_S),
                default=_DEFAULT_LOSS_EXTRAPOLATION_S,
                minimum=0.0,
                maximum=5.0,
            ),
            loss_extrapolation_gain=_bounded_float(
                getattr(config, "attention_servo_loss_extrapolation_gain", _DEFAULT_LOSS_EXTRAPOLATION_GAIN),
                default=_DEFAULT_LOSS_EXTRAPOLATION_GAIN,
                minimum=0.0,
                maximum=4.0,
            ),
            min_confidence=_bounded_float(
                getattr(config, "attention_servo_min_confidence", _DEFAULT_MIN_CONFIDENCE),
                default=_DEFAULT_MIN_CONFIDENCE,
                minimum=0.0,
                maximum=1.0,
            ),
            hold_min_confidence=_bounded_float(
                getattr(config, "attention_servo_hold_min_confidence", _DEFAULT_MIN_CONFIDENCE),
                default=_DEFAULT_MIN_CONFIDENCE,
                minimum=0.0,
                maximum=_bounded_float(
                    getattr(config, "attention_servo_min_confidence", _DEFAULT_MIN_CONFIDENCE),
                    default=_DEFAULT_MIN_CONFIDENCE,
                    minimum=0.0,
                    maximum=1.0,
                ),
            ),
            deadband=_bounded_float(
                getattr(config, "attention_servo_deadband", _DEFAULT_DEADBAND),
                default=_DEFAULT_DEADBAND,
                minimum=0.0,
                maximum=0.3,
            ),
            min_pulse_width_us=min_pulse,
            center_pulse_width_us=center_pulse,
            max_pulse_width_us=max_pulse,
            max_step_us=_bounded_int(
                getattr(config, "attention_servo_max_step_us", _DEFAULT_MAX_STEP_US),
                default=_DEFAULT_MAX_STEP_US,
                minimum=1,
                maximum=max(1, max_pulse - min_pulse),
            ),
            target_smoothing_s=_bounded_float(
                getattr(config, "attention_servo_target_smoothing_s", _DEFAULT_TARGET_SMOOTHING_S),
                default=_DEFAULT_TARGET_SMOOTHING_S,
                minimum=0.0,
                maximum=5.0,
            ),
            max_velocity_us_per_s=_bounded_float(
                getattr(config, "attention_servo_max_velocity_us_per_s", _DEFAULT_MAX_VELOCITY_US_PER_S),
                default=_DEFAULT_MAX_VELOCITY_US_PER_S,
                minimum=1.0,
                maximum=1000.0,
            ),
            max_acceleration_us_per_s2=_bounded_float(
                getattr(
                    config,
                    "attention_servo_max_acceleration_us_per_s2",
                    _DEFAULT_MAX_ACCELERATION_US_PER_S2,
                ),
                default=_DEFAULT_MAX_ACCELERATION_US_PER_S2,
                minimum=1.0,
                maximum=10000.0,
            ),
            max_jerk_us_per_s3=_bounded_float(
                getattr(config, "attention_servo_max_jerk_us_per_s3", _DEFAULT_MAX_JERK_US_PER_S3),
                default=_DEFAULT_MAX_JERK_US_PER_S3,
                minimum=1.0,
                maximum=100000.0,
            ),
            rest_max_velocity_us_per_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_rest_max_velocity_us_per_s",
                    _DEFAULT_REST_MAX_VELOCITY_US_PER_S,
                ),
                default=_DEFAULT_REST_MAX_VELOCITY_US_PER_S,
                minimum=1.0,
                maximum=1000.0,
            ),
            rest_max_acceleration_us_per_s2=_bounded_float(
                getattr(
                    config,
                    "attention_servo_rest_max_acceleration_us_per_s2",
                    _DEFAULT_REST_MAX_ACCELERATION_US_PER_S2,
                ),
                default=_DEFAULT_REST_MAX_ACCELERATION_US_PER_S2,
                minimum=1.0,
                maximum=10000.0,
            ),
            rest_max_jerk_us_per_s3=_bounded_float(
                getattr(
                    config,
                    "attention_servo_rest_max_jerk_us_per_s3",
                    _DEFAULT_REST_MAX_JERK_US_PER_S3,
                ),
                default=_DEFAULT_REST_MAX_JERK_US_PER_S3,
                minimum=1.0,
                maximum=100000.0,
            ),
            min_command_delta_us=_bounded_int(
                getattr(config, "attention_servo_min_command_delta_us", _DEFAULT_MIN_COMMAND_DELTA_US),
                default=_DEFAULT_MIN_COMMAND_DELTA_US,
                minimum=1,
                maximum=max(1, max_pulse - min_pulse),
            ),
            visible_retarget_tolerance_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_visible_retarget_tolerance_us",
                    _DEFAULT_VISIBLE_RETARGET_TOLERANCE_US,
                ),
                default=_DEFAULT_VISIBLE_RETARGET_TOLERANCE_US,
                minimum=0,
                maximum=max(1, max_pulse - min_pulse),
            ),
            reference_interval_s=_bounded_float(
                getattr(config, "display_attention_refresh_interval_s", _DEFAULT_REFERENCE_INTERVAL_S),
                default=_DEFAULT_REFERENCE_INTERVAL_S,
                minimum=0.05,
                maximum=2.0,
            ),
            soft_limit_margin_us=_bounded_int(
                getattr(config, "attention_servo_soft_limit_margin_us", _DEFAULT_SOFT_LIMIT_MARGIN_US),
                default=_DEFAULT_SOFT_LIMIT_MARGIN_US,
                minimum=0,
                maximum=max(0, max_pulse - min_pulse),
            ),
            idle_release_s=_bounded_float(
                getattr(config, "attention_servo_idle_release_s", _DEFAULT_IDLE_RELEASE_S),
                default=_DEFAULT_IDLE_RELEASE_S,
                minimum=0.0,
                maximum=10.0,
            ),
            settled_release_s=_bounded_float(
                getattr(config, "attention_servo_settled_release_s", _DEFAULT_SETTLED_RELEASE_S),
                default=_DEFAULT_SETTLED_RELEASE_S,
                minimum=0.0,
                maximum=10.0,
            ),
            follow_exit_only=bool(
                getattr(config, "attention_servo_follow_exit_only", _DEFAULT_FOLLOW_EXIT_ONLY)
            ),
            visible_recenter_interval_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_visible_recenter_interval_s",
                    _DEFAULT_VISIBLE_RECENTER_INTERVAL_S,
                ),
                default=_DEFAULT_VISIBLE_RECENTER_INTERVAL_S,
                minimum=0.0,
                maximum=600.0,
            ),
            visible_recenter_center_tolerance=_bounded_float(
                getattr(
                    config,
                    "attention_servo_visible_recenter_center_tolerance",
                    _DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE,
                ),
                default=_DEFAULT_VISIBLE_RECENTER_CENTER_TOLERANCE,
                minimum=0.0,
                maximum=0.3,
            ),
            mechanical_range_degrees=mechanical_range_degrees,
            exit_follow_max_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_follow_max_degrees",
                    _DEFAULT_EXIT_FOLLOW_MAX_DEGREES,
                ),
                default=_DEFAULT_EXIT_FOLLOW_MAX_DEGREES,
                minimum=0.0,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            exit_activation_delay_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_activation_delay_s",
                    _DEFAULT_EXIT_ACTIVATION_DELAY_S,
                ),
                default=_DEFAULT_EXIT_ACTIVATION_DELAY_S,
                minimum=0.0,
                maximum=max(0.0, target_hold_s),
            ),
            exit_settle_hold_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_settle_hold_s",
                    _DEFAULT_EXIT_SETTLE_HOLD_S,
                ),
                default=_DEFAULT_EXIT_SETTLE_HOLD_S,
                minimum=0.0,
                maximum=10.0,
            ),
            exit_reacquire_center_tolerance=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_reacquire_center_tolerance",
                    _DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE,
                ),
                default=_DEFAULT_EXIT_REACQUIRE_CENTER_TOLERANCE,
                minimum=0.0,
                maximum=0.3,
            ),
            exit_visible_edge_threshold=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_visible_edge_threshold",
                    _DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD,
                ),
                default=_DEFAULT_EXIT_VISIBLE_EDGE_THRESHOLD,
                minimum=0.55,
                maximum=0.95,
            ),
            exit_visible_box_edge_threshold=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_visible_box_edge_threshold",
                    _DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD,
                ),
                default=_DEFAULT_EXIT_VISIBLE_BOX_EDGE_THRESHOLD,
                minimum=0.75,
                maximum=0.99,
            ),
            exit_cooldown_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_exit_cooldown_s",
                    _DEFAULT_EXIT_COOLDOWN_S,
                ),
                default=_DEFAULT_EXIT_COOLDOWN_S,
                minimum=0.0,
                maximum=300.0,
            ),
            continuous_max_speed_degrees_per_s=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_max_speed_degrees_per_s",
                    _DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S,
                ),
                default=_DEFAULT_CONTINUOUS_MAX_SPEED_DEGREES_PER_S,
                minimum=1.0,
                maximum=720.0,
            ),
            continuous_slow_zone_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_slow_zone_degrees",
                    _DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES,
                ),
                default=_DEFAULT_CONTINUOUS_SLOW_ZONE_DEGREES,
                minimum=0.5,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            continuous_stop_tolerance_degrees=_bounded_float(
                getattr(
                    config,
                    "attention_servo_continuous_stop_tolerance_degrees",
                    _DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES,
                ),
                default=_DEFAULT_CONTINUOUS_STOP_TOLERANCE_DEGREES,
                minimum=0.0,
                maximum=max(1.0, mechanical_range_degrees * 0.5),
            ),
            continuous_min_speed_pulse_delta_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_continuous_min_speed_pulse_delta_us",
                    _DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US,
                ),
                default=_DEFAULT_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US,
                minimum=0,
                maximum=max(1, min(center_pulse - min_pulse, max_pulse - center_pulse)),
            ),
            continuous_max_speed_pulse_delta_us=_bounded_int(
                getattr(
                    config,
                    "attention_servo_continuous_max_speed_pulse_delta_us",
                    _DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US,
                ),
                default=_DEFAULT_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US,
                minimum=0,
                maximum=max(1, min(center_pulse - min_pulse, max_pulse - center_pulse)),
            ),
        )
