"""Normalize and validate the immutable Twinr runtime config snapshot.

Purpose and boundaries:
- Own post-construction normalization for ``TwinrConfig``.
- Clamp and validate values without changing the public dataclass surface.
- Emit only in-memory mutations through ``object.__setattr__`` on the frozen config.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .constants import (
    SUPPORTED_DISPLAY_DRIVERS,
    SUPPORTED_DISPLAY_LAYOUTS,
    DEFAULT_OPENAI_MAIN_MODEL,
)
from .parsing import (
    _normalize_model_setting,
    _parse_attention_servo_control_mode,
    _parse_attention_servo_driver,
    _parse_camera_host_mode,
    _parse_optional_url,
)
from .normalization_updates import (
    apply_attention_servo_updates,
    apply_display_updates,
    apply_general_updates,
)

if TYPE_CHECKING:
    from .schema import TwinrConfig


def normalize_twinr_config(config: "TwinrConfig") -> None:
    """Normalize derived long-term-memory mode fields after construction."""

    normalized_mode = (
        str(config.long_term_memory_mode or "local_first").strip().lower()
        or "local_first"
    )
    normalized_display_driver = (
        str(config.display_driver or "hdmi_fbdev").strip().lower() or "hdmi_fbdev"
    )
    normalized_display_layout = (
        str(config.display_layout or "default").strip().lower() or "default"
    )
    if normalized_display_layout == "debug_face":
        normalized_display_layout = "debug_log"
    if normalized_display_driver not in SUPPORTED_DISPLAY_DRIVERS:
        raise ValueError(
            "display_driver must be one of: " + ", ".join(SUPPORTED_DISPLAY_DRIVERS)
        )
    if normalized_display_layout not in SUPPORTED_DISPLAY_LAYOUTS:
        raise ValueError(
            "display_layout must be one of: " + ", ".join(SUPPORTED_DISPLAY_LAYOUTS)
        )
    normalized_display_busy_timeout_s = float(config.display_busy_timeout_s)
    if not math.isfinite(normalized_display_busy_timeout_s):
        raise ValueError("display_busy_timeout_s must be finite")
    normalized_display_busy_timeout_s = max(0.1, normalized_display_busy_timeout_s)
    normalized_display_face_cue_ttl_s = float(config.display_face_cue_ttl_s)
    if not math.isfinite(normalized_display_face_cue_ttl_s):
        raise ValueError("display_face_cue_ttl_s must be finite")
    normalized_display_face_cue_ttl_s = max(0.1, normalized_display_face_cue_ttl_s)
    normalized_display_emoji_cue_ttl_s = float(config.display_emoji_cue_ttl_s)
    if not math.isfinite(normalized_display_emoji_cue_ttl_s):
        raise ValueError("display_emoji_cue_ttl_s must be finite")
    normalized_display_emoji_cue_ttl_s = max(0.1, normalized_display_emoji_cue_ttl_s)
    normalized_display_ambient_impulse_ttl_s = float(
        config.display_ambient_impulse_ttl_s
    )
    if not math.isfinite(normalized_display_ambient_impulse_ttl_s):
        raise ValueError("display_ambient_impulse_ttl_s must be finite")
    normalized_display_ambient_impulse_ttl_s = max(
        0.1, normalized_display_ambient_impulse_ttl_s
    )
    normalized_default_model = _normalize_model_setting(
        config.default_model,
        fallback=DEFAULT_OPENAI_MAIN_MODEL,
    )
    normalized_streaming_first_word_model = _normalize_model_setting(
        config.streaming_first_word_model,
        fallback=normalized_default_model,
    )
    normalized_streaming_supervisor_model = _normalize_model_setting(
        config.streaming_supervisor_model,
        fallback=normalized_default_model,
    )
    normalized_streaming_specialist_model = _normalize_model_setting(
        config.streaming_specialist_model,
        fallback=normalized_default_model,
    )
    normalized_conversation_closure_model = _normalize_model_setting(
        config.conversation_closure_model,
        fallback=normalized_default_model,
    )
    normalized_openai_search_model = _normalize_model_setting(
        config.openai_search_model,
        fallback=normalized_default_model,
    )
    normalized_display_reserve_generation_model = _normalize_model_setting(
        config.display_reserve_generation_model,
        fallback=normalized_default_model,
    )
    normalized_display_reserve_generation_reasoning_effort = (
        str(config.display_reserve_generation_reasoning_effort or "low").strip().lower()
        or "low"
    )
    normalized_display_reserve_generation_timeout_seconds = float(
        config.display_reserve_generation_timeout_seconds
    )
    if not math.isfinite(normalized_display_reserve_generation_timeout_seconds):
        raise ValueError("display_reserve_generation_timeout_seconds must be finite")
    normalized_display_reserve_generation_timeout_seconds = max(
        1.0,
        normalized_display_reserve_generation_timeout_seconds,
    )
    normalized_display_reserve_generation_max_output_tokens = max(
        128,
        int(config.display_reserve_generation_max_output_tokens),
    )
    normalized_display_reserve_generation_batch_size = max(
        1,
        int(config.display_reserve_generation_batch_size),
    )
    normalized_display_reserve_generation_variants_per_candidate = max(
        1,
        int(config.display_reserve_generation_variants_per_candidate),
    )
    normalized_display_reserve_bus_candidate_limit = max(
        1, int(config.display_reserve_bus_candidate_limit)
    )
    normalized_display_reserve_bus_items_per_day = max(
        1, int(config.display_reserve_bus_items_per_day)
    )
    normalized_display_reserve_bus_topic_gap = max(
        0, int(config.display_reserve_bus_topic_gap)
    )
    normalized_display_reserve_bus_learning_window_days = float(
        config.display_reserve_bus_learning_window_days
    )
    if not math.isfinite(normalized_display_reserve_bus_learning_window_days):
        raise ValueError("display_reserve_bus_learning_window_days must be finite")
    normalized_display_reserve_bus_learning_window_days = max(
        3.0,
        normalized_display_reserve_bus_learning_window_days,
    )
    normalized_display_reserve_bus_learning_half_life_days = float(
        config.display_reserve_bus_learning_half_life_days
    )
    if not math.isfinite(normalized_display_reserve_bus_learning_half_life_days):
        raise ValueError("display_reserve_bus_learning_half_life_days must be finite")
    normalized_display_reserve_bus_learning_half_life_days = max(
        1.0,
        normalized_display_reserve_bus_learning_half_life_days,
    )
    normalized_display_reserve_bus_reflection_candidate_limit = max(
        1,
        int(config.display_reserve_bus_reflection_candidate_limit),
    )
    normalized_display_reserve_bus_reflection_max_age_days = float(
        config.display_reserve_bus_reflection_max_age_days
    )
    if not math.isfinite(normalized_display_reserve_bus_reflection_max_age_days):
        raise ValueError("display_reserve_bus_reflection_max_age_days must be finite")
    normalized_display_reserve_bus_reflection_max_age_days = max(
        1.0,
        normalized_display_reserve_bus_reflection_max_age_days,
    )
    normalized_display_reserve_bus_min_hold_s = float(
        config.display_reserve_bus_min_hold_s
    )
    if not math.isfinite(normalized_display_reserve_bus_min_hold_s):
        raise ValueError("display_reserve_bus_min_hold_s must be finite")
    normalized_display_reserve_bus_min_hold_s = max(
        60.0, normalized_display_reserve_bus_min_hold_s
    )
    normalized_display_reserve_bus_base_hold_s = float(
        config.display_reserve_bus_base_hold_s
    )
    if not math.isfinite(normalized_display_reserve_bus_base_hold_s):
        raise ValueError("display_reserve_bus_base_hold_s must be finite")
    normalized_display_reserve_bus_base_hold_s = max(
        normalized_display_reserve_bus_min_hold_s,
        normalized_display_reserve_bus_base_hold_s,
    )
    normalized_display_reserve_bus_max_hold_s = float(
        config.display_reserve_bus_max_hold_s
    )
    if not math.isfinite(normalized_display_reserve_bus_max_hold_s):
        raise ValueError("display_reserve_bus_max_hold_s must be finite")
    normalized_display_reserve_bus_max_hold_s = max(
        normalized_display_reserve_bus_base_hold_s,
        normalized_display_reserve_bus_max_hold_s,
    )
    normalized_display_reserve_bus_nightly_poll_interval_s = float(
        config.display_reserve_bus_nightly_poll_interval_s
    )
    if not math.isfinite(normalized_display_reserve_bus_nightly_poll_interval_s):
        raise ValueError("display_reserve_bus_nightly_poll_interval_s must be finite")
    normalized_display_reserve_bus_nightly_poll_interval_s = max(
        30.0,
        normalized_display_reserve_bus_nightly_poll_interval_s,
    )
    normalized_display_attention_refresh_interval_s = float(
        config.display_attention_refresh_interval_s
    )
    if not math.isfinite(normalized_display_attention_refresh_interval_s):
        raise ValueError("display_attention_refresh_interval_s must be finite")
    normalized_display_attention_refresh_interval_s = max(
        0.0, normalized_display_attention_refresh_interval_s
    )
    normalized_display_attention_session_focus_hold_s = float(
        config.display_attention_session_focus_hold_s
    )
    if not math.isfinite(normalized_display_attention_session_focus_hold_s):
        raise ValueError("display_attention_session_focus_hold_s must be finite")
    normalized_display_attention_session_focus_hold_s = max(
        0.5, normalized_display_attention_session_focus_hold_s
    )
    normalized_attention_servo_target_hold_s = float(
        config.attention_servo_target_hold_s
    )
    if not math.isfinite(normalized_attention_servo_target_hold_s):
        raise ValueError("attention_servo_target_hold_s must be finite")
    normalized_attention_servo_target_hold_s = max(
        0.0, normalized_attention_servo_target_hold_s
    )
    normalized_attention_servo_loss_extrapolation_s = float(
        config.attention_servo_loss_extrapolation_s
    )
    if not math.isfinite(normalized_attention_servo_loss_extrapolation_s):
        raise ValueError("attention_servo_loss_extrapolation_s must be finite")
    normalized_attention_servo_loss_extrapolation_s = max(
        0.0, normalized_attention_servo_loss_extrapolation_s
    )
    normalized_attention_servo_loss_extrapolation_gain = float(
        config.attention_servo_loss_extrapolation_gain
    )
    if not math.isfinite(normalized_attention_servo_loss_extrapolation_gain):
        raise ValueError("attention_servo_loss_extrapolation_gain must be finite")
    normalized_attention_servo_loss_extrapolation_gain = max(
        0.0,
        normalized_attention_servo_loss_extrapolation_gain,
    )
    normalized_attention_servo_driver = _parse_attention_servo_driver(
        config.attention_servo_driver, "auto"
    )
    normalized_attention_servo_control_mode = _parse_attention_servo_control_mode(
        config.attention_servo_control_mode,
        "position",
    )
    normalized_attention_servo_maestro_device = (
        None
        if config.attention_servo_maestro_device is None
        else (str(config.attention_servo_maestro_device).strip() or None)
    )
    normalized_attention_servo_maestro_channel = (
        None
        if config.attention_servo_maestro_channel is None
        else max(0, min(23, int(config.attention_servo_maestro_channel)))
    )
    normalized_attention_servo_peer_base_url = (
        None
        if config.attention_servo_peer_base_url is None
        else (str(config.attention_servo_peer_base_url).strip().rstrip("/") or None)
    )
    normalized_attention_servo_peer_timeout_s = float(
        config.attention_servo_peer_timeout_s
    )
    if not math.isfinite(normalized_attention_servo_peer_timeout_s):
        raise ValueError("attention_servo_peer_timeout_s must be finite")
    normalized_attention_servo_peer_timeout_s = max(
        0.1, normalized_attention_servo_peer_timeout_s
    )
    normalized_attention_servo_state_path = (
        str(
            config.attention_servo_state_path or "state/attention_servo_state.json"
        ).strip()
        or "state/attention_servo_state.json"
    )
    normalized_attention_servo_estimated_zero_max_uncertainty_degrees = float(
        config.attention_servo_estimated_zero_max_uncertainty_degrees
    )
    if not math.isfinite(
        normalized_attention_servo_estimated_zero_max_uncertainty_degrees
    ):
        raise ValueError(
            "attention_servo_estimated_zero_max_uncertainty_degrees must be finite"
        )
    normalized_attention_servo_estimated_zero_max_uncertainty_degrees = max(
        0.0,
        min(180.0, normalized_attention_servo_estimated_zero_max_uncertainty_degrees),
    )
    normalized_attention_servo_estimated_zero_settle_tolerance_degrees = float(
        config.attention_servo_estimated_zero_settle_tolerance_degrees
    )
    if not math.isfinite(
        normalized_attention_servo_estimated_zero_settle_tolerance_degrees
    ):
        raise ValueError(
            "attention_servo_estimated_zero_settle_tolerance_degrees must be finite"
        )
    normalized_attention_servo_estimated_zero_settle_tolerance_degrees = max(
        0.0,
        min(180.0, normalized_attention_servo_estimated_zero_settle_tolerance_degrees),
    )
    normalized_attention_servo_estimated_zero_speed_scale = float(
        config.attention_servo_estimated_zero_speed_scale
    )
    if not math.isfinite(normalized_attention_servo_estimated_zero_speed_scale):
        raise ValueError("attention_servo_estimated_zero_speed_scale must be finite")
    normalized_attention_servo_estimated_zero_speed_scale = max(
        0.0,
        min(1.0, normalized_attention_servo_estimated_zero_speed_scale),
    )
    normalized_attention_servo_estimated_zero_move_pulse_delta_us = int(
        config.attention_servo_estimated_zero_move_pulse_delta_us
    )
    normalized_attention_servo_estimated_zero_move_pulse_delta_us = max(
        0,
        min(500, normalized_attention_servo_estimated_zero_move_pulse_delta_us),
    )
    normalized_attention_servo_estimated_zero_move_period_s = float(
        config.attention_servo_estimated_zero_move_period_s
    )
    if not math.isfinite(normalized_attention_servo_estimated_zero_move_period_s):
        raise ValueError("attention_servo_estimated_zero_move_period_s must be finite")
    normalized_attention_servo_estimated_zero_move_period_s = max(
        0.05,
        min(10.0, normalized_attention_servo_estimated_zero_move_period_s),
    )
    normalized_attention_servo_estimated_zero_move_duty_cycle = float(
        config.attention_servo_estimated_zero_move_duty_cycle
    )
    if not math.isfinite(normalized_attention_servo_estimated_zero_move_duty_cycle):
        raise ValueError(
            "attention_servo_estimated_zero_move_duty_cycle must be finite"
        )
    normalized_attention_servo_estimated_zero_move_duty_cycle = max(
        0.05,
        min(1.0, normalized_attention_servo_estimated_zero_move_duty_cycle),
    )
    normalized_attention_servo_continuous_return_to_zero_after_s = float(
        config.attention_servo_continuous_return_to_zero_after_s
    )
    if not math.isfinite(normalized_attention_servo_continuous_return_to_zero_after_s):
        raise ValueError(
            "attention_servo_continuous_return_to_zero_after_s must be finite"
        )
    normalized_attention_servo_continuous_return_to_zero_after_s = max(
        0.0,
        min(3600.0, normalized_attention_servo_continuous_return_to_zero_after_s),
    )
    normalized_attention_servo_min_confidence = float(
        config.attention_servo_min_confidence
    )
    if not math.isfinite(normalized_attention_servo_min_confidence):
        raise ValueError("attention_servo_min_confidence must be finite")
    normalized_attention_servo_min_confidence = max(
        0.0, min(1.0, normalized_attention_servo_min_confidence)
    )
    normalized_attention_servo_hold_min_confidence = float(
        config.attention_servo_hold_min_confidence
    )
    if not math.isfinite(normalized_attention_servo_hold_min_confidence):
        raise ValueError("attention_servo_hold_min_confidence must be finite")
    normalized_attention_servo_hold_min_confidence = max(
        0.0,
        min(
            normalized_attention_servo_min_confidence,
            normalized_attention_servo_hold_min_confidence,
        ),
    )
    normalized_attention_servo_deadband = float(config.attention_servo_deadband)
    if not math.isfinite(normalized_attention_servo_deadband):
        raise ValueError("attention_servo_deadband must be finite")
    normalized_attention_servo_deadband = max(
        0.0, min(0.3, normalized_attention_servo_deadband)
    )
    normalized_attention_servo_min_pulse_width_us = max(
        500, int(config.attention_servo_min_pulse_width_us)
    )
    normalized_attention_servo_max_pulse_width_us = max(
        500, int(config.attention_servo_max_pulse_width_us)
    )
    if (
        normalized_attention_servo_max_pulse_width_us
        < normalized_attention_servo_min_pulse_width_us
    ):
        (
            normalized_attention_servo_min_pulse_width_us,
            normalized_attention_servo_max_pulse_width_us,
        ) = (
            normalized_attention_servo_max_pulse_width_us,
            normalized_attention_servo_min_pulse_width_us,
        )
    normalized_attention_servo_center_pulse_width_us = int(
        config.attention_servo_center_pulse_width_us
    )
    normalized_attention_servo_center_pulse_width_us = max(
        normalized_attention_servo_min_pulse_width_us,
        min(
            normalized_attention_servo_max_pulse_width_us,
            normalized_attention_servo_center_pulse_width_us,
        ),
    )
    normalized_attention_servo_max_step_us = max(
        1,
        min(
            int(config.attention_servo_max_step_us),
            normalized_attention_servo_max_pulse_width_us
            - normalized_attention_servo_min_pulse_width_us,
        ),
    )
    normalized_attention_servo_target_smoothing_s = float(
        config.attention_servo_target_smoothing_s
    )
    if not math.isfinite(normalized_attention_servo_target_smoothing_s):
        raise ValueError("attention_servo_target_smoothing_s must be finite")
    normalized_attention_servo_target_smoothing_s = max(
        0.0, normalized_attention_servo_target_smoothing_s
    )
    normalized_attention_servo_max_velocity_us_per_s = float(
        config.attention_servo_max_velocity_us_per_s
    )
    if not math.isfinite(normalized_attention_servo_max_velocity_us_per_s):
        raise ValueError("attention_servo_max_velocity_us_per_s must be finite")
    normalized_attention_servo_max_velocity_us_per_s = max(
        1.0,
        normalized_attention_servo_max_velocity_us_per_s,
    )
    normalized_attention_servo_max_acceleration_us_per_s2 = float(
        config.attention_servo_max_acceleration_us_per_s2
    )
    if not math.isfinite(normalized_attention_servo_max_acceleration_us_per_s2):
        raise ValueError("attention_servo_max_acceleration_us_per_s2 must be finite")
    normalized_attention_servo_max_acceleration_us_per_s2 = max(
        1.0,
        normalized_attention_servo_max_acceleration_us_per_s2,
    )
    normalized_attention_servo_max_jerk_us_per_s3 = float(
        config.attention_servo_max_jerk_us_per_s3
    )
    if not math.isfinite(normalized_attention_servo_max_jerk_us_per_s3):
        raise ValueError("attention_servo_max_jerk_us_per_s3 must be finite")
    normalized_attention_servo_max_jerk_us_per_s3 = max(
        1.0,
        normalized_attention_servo_max_jerk_us_per_s3,
    )
    normalized_attention_servo_rest_max_velocity_us_per_s = float(
        config.attention_servo_rest_max_velocity_us_per_s
    )
    if not math.isfinite(normalized_attention_servo_rest_max_velocity_us_per_s):
        raise ValueError("attention_servo_rest_max_velocity_us_per_s must be finite")
    normalized_attention_servo_rest_max_velocity_us_per_s = max(
        1.0,
        normalized_attention_servo_rest_max_velocity_us_per_s,
    )
    normalized_attention_servo_rest_max_acceleration_us_per_s2 = float(
        config.attention_servo_rest_max_acceleration_us_per_s2
    )
    if not math.isfinite(normalized_attention_servo_rest_max_acceleration_us_per_s2):
        raise ValueError(
            "attention_servo_rest_max_acceleration_us_per_s2 must be finite"
        )
    normalized_attention_servo_rest_max_acceleration_us_per_s2 = max(
        1.0,
        normalized_attention_servo_rest_max_acceleration_us_per_s2,
    )
    normalized_attention_servo_rest_max_jerk_us_per_s3 = float(
        config.attention_servo_rest_max_jerk_us_per_s3
    )
    if not math.isfinite(normalized_attention_servo_rest_max_jerk_us_per_s3):
        raise ValueError("attention_servo_rest_max_jerk_us_per_s3 must be finite")
    normalized_attention_servo_rest_max_jerk_us_per_s3 = max(
        1.0,
        normalized_attention_servo_rest_max_jerk_us_per_s3,
    )
    normalized_attention_servo_min_command_delta_us = max(
        1,
        min(
            int(config.attention_servo_min_command_delta_us),
            normalized_attention_servo_max_pulse_width_us
            - normalized_attention_servo_min_pulse_width_us,
        ),
    )
    normalized_attention_servo_visible_retarget_tolerance_us = max(
        0,
        min(
            int(config.attention_servo_visible_retarget_tolerance_us),
            normalized_attention_servo_max_pulse_width_us
            - normalized_attention_servo_min_pulse_width_us,
        ),
    )
    normalized_attention_servo_soft_limit_margin_us = max(
        0,
        min(
            int(config.attention_servo_soft_limit_margin_us),
            normalized_attention_servo_max_pulse_width_us
            - normalized_attention_servo_min_pulse_width_us,
        ),
    )
    normalized_attention_servo_idle_release_s = float(
        config.attention_servo_idle_release_s
    )
    if not math.isfinite(normalized_attention_servo_idle_release_s):
        raise ValueError("attention_servo_idle_release_s must be finite")
    normalized_attention_servo_idle_release_s = max(
        0.0, normalized_attention_servo_idle_release_s
    )
    normalized_attention_servo_settled_release_s = float(
        config.attention_servo_settled_release_s
    )
    if not math.isfinite(normalized_attention_servo_settled_release_s):
        raise ValueError("attention_servo_settled_release_s must be finite")
    normalized_attention_servo_settled_release_s = max(
        0.0, normalized_attention_servo_settled_release_s
    )
    normalized_attention_servo_visible_recenter_interval_s = float(
        config.attention_servo_visible_recenter_interval_s
    )
    if not math.isfinite(normalized_attention_servo_visible_recenter_interval_s):
        raise ValueError("attention_servo_visible_recenter_interval_s must be finite")
    normalized_attention_servo_visible_recenter_interval_s = max(
        0.0,
        normalized_attention_servo_visible_recenter_interval_s,
    )
    normalized_attention_servo_visible_recenter_center_tolerance = float(
        config.attention_servo_visible_recenter_center_tolerance
    )
    if not math.isfinite(normalized_attention_servo_visible_recenter_center_tolerance):
        raise ValueError(
            "attention_servo_visible_recenter_center_tolerance must be finite"
        )
    normalized_attention_servo_visible_recenter_center_tolerance = max(
        0.0,
        min(0.3, normalized_attention_servo_visible_recenter_center_tolerance),
    )
    normalized_attention_servo_mechanical_range_degrees = float(
        config.attention_servo_mechanical_range_degrees
    )
    if not math.isfinite(normalized_attention_servo_mechanical_range_degrees):
        raise ValueError("attention_servo_mechanical_range_degrees must be finite")
    normalized_attention_servo_mechanical_range_degrees = max(
        30.0,
        min(360.0, normalized_attention_servo_mechanical_range_degrees),
    )
    normalized_attention_servo_exit_follow_max_degrees = float(
        config.attention_servo_exit_follow_max_degrees
    )
    if not math.isfinite(normalized_attention_servo_exit_follow_max_degrees):
        raise ValueError("attention_servo_exit_follow_max_degrees must be finite")
    normalized_attention_servo_exit_follow_max_degrees = max(
        0.0,
        min(
            normalized_attention_servo_mechanical_range_degrees * 0.5,
            normalized_attention_servo_exit_follow_max_degrees,
        ),
    )
    normalized_attention_servo_exit_activation_delay_s = float(
        config.attention_servo_exit_activation_delay_s
    )
    if not math.isfinite(normalized_attention_servo_exit_activation_delay_s):
        raise ValueError("attention_servo_exit_activation_delay_s must be finite")
    normalized_attention_servo_exit_activation_delay_s = max(
        0.0,
        min(
            normalized_attention_servo_target_hold_s,
            normalized_attention_servo_exit_activation_delay_s,
        ),
    )
    normalized_attention_servo_exit_settle_hold_s = float(
        config.attention_servo_exit_settle_hold_s
    )
    if not math.isfinite(normalized_attention_servo_exit_settle_hold_s):
        raise ValueError("attention_servo_exit_settle_hold_s must be finite")
    normalized_attention_servo_exit_settle_hold_s = max(
        0.0, normalized_attention_servo_exit_settle_hold_s
    )
    normalized_attention_servo_exit_reacquire_center_tolerance = float(
        config.attention_servo_exit_reacquire_center_tolerance
    )
    if not math.isfinite(normalized_attention_servo_exit_reacquire_center_tolerance):
        raise ValueError(
            "attention_servo_exit_reacquire_center_tolerance must be finite"
        )
    normalized_attention_servo_exit_reacquire_center_tolerance = max(
        0.0,
        min(0.3, normalized_attention_servo_exit_reacquire_center_tolerance),
    )
    normalized_attention_servo_exit_visible_edge_threshold = float(
        config.attention_servo_exit_visible_edge_threshold
    )
    if not math.isfinite(normalized_attention_servo_exit_visible_edge_threshold):
        raise ValueError("attention_servo_exit_visible_edge_threshold must be finite")
    normalized_attention_servo_exit_visible_edge_threshold = max(
        0.55,
        min(0.95, normalized_attention_servo_exit_visible_edge_threshold),
    )
    normalized_attention_servo_exit_visible_box_edge_threshold = float(
        config.attention_servo_exit_visible_box_edge_threshold
    )
    if not math.isfinite(normalized_attention_servo_exit_visible_box_edge_threshold):
        raise ValueError(
            "attention_servo_exit_visible_box_edge_threshold must be finite"
        )
    normalized_attention_servo_exit_visible_box_edge_threshold = max(
        0.75,
        min(0.99, normalized_attention_servo_exit_visible_box_edge_threshold),
    )
    normalized_attention_servo_exit_cooldown_s = float(
        config.attention_servo_exit_cooldown_s
    )
    if not math.isfinite(normalized_attention_servo_exit_cooldown_s):
        raise ValueError("attention_servo_exit_cooldown_s must be finite")
    normalized_attention_servo_exit_cooldown_s = max(
        0.0, normalized_attention_servo_exit_cooldown_s
    )
    normalized_attention_servo_continuous_max_speed_degrees_per_s = float(
        config.attention_servo_continuous_max_speed_degrees_per_s
    )
    if not math.isfinite(normalized_attention_servo_continuous_max_speed_degrees_per_s):
        raise ValueError(
            "attention_servo_continuous_max_speed_degrees_per_s must be finite"
        )
    normalized_attention_servo_continuous_max_speed_degrees_per_s = max(
        1.0,
        normalized_attention_servo_continuous_max_speed_degrees_per_s,
    )
    normalized_attention_servo_continuous_slow_zone_degrees = float(
        config.attention_servo_continuous_slow_zone_degrees
    )
    if not math.isfinite(normalized_attention_servo_continuous_slow_zone_degrees):
        raise ValueError("attention_servo_continuous_slow_zone_degrees must be finite")
    normalized_attention_servo_continuous_slow_zone_degrees = max(
        0.5,
        min(
            normalized_attention_servo_mechanical_range_degrees * 0.5,
            normalized_attention_servo_continuous_slow_zone_degrees,
        ),
    )
    normalized_attention_servo_continuous_stop_tolerance_degrees = float(
        config.attention_servo_continuous_stop_tolerance_degrees
    )
    if not math.isfinite(normalized_attention_servo_continuous_stop_tolerance_degrees):
        raise ValueError(
            "attention_servo_continuous_stop_tolerance_degrees must be finite"
        )
    normalized_attention_servo_continuous_stop_tolerance_degrees = max(
        0.0,
        min(
            normalized_attention_servo_continuous_slow_zone_degrees,
            normalized_attention_servo_continuous_stop_tolerance_degrees,
        ),
    )
    continuous_available_delta_us = max(
        1,
        min(
            normalized_attention_servo_center_pulse_width_us
            - normalized_attention_servo_min_pulse_width_us,
            normalized_attention_servo_max_pulse_width_us
            - normalized_attention_servo_center_pulse_width_us,
        ),
    )
    normalized_attention_servo_continuous_min_speed_pulse_delta_us = max(
        0,
        min(
            int(config.attention_servo_continuous_min_speed_pulse_delta_us),
            continuous_available_delta_us,
        ),
    )
    normalized_attention_servo_continuous_max_speed_pulse_delta_us = max(
        normalized_attention_servo_continuous_min_speed_pulse_delta_us,
        min(
            int(config.attention_servo_continuous_max_speed_pulse_delta_us),
            continuous_available_delta_us,
        ),
    )
    normalized_display_presentation_ttl_s = float(config.display_presentation_ttl_s)
    if not math.isfinite(normalized_display_presentation_ttl_s):
        raise ValueError("display_presentation_ttl_s must be finite")
    normalized_display_presentation_ttl_s = max(
        0.1, normalized_display_presentation_ttl_s
    )
    normalized_display_news_ticker_refresh_interval_s = float(
        config.display_news_ticker_refresh_interval_s
    )
    if not math.isfinite(normalized_display_news_ticker_refresh_interval_s):
        raise ValueError("display_news_ticker_refresh_interval_s must be finite")
    normalized_display_news_ticker_refresh_interval_s = max(
        30.0, normalized_display_news_ticker_refresh_interval_s
    )
    normalized_display_news_ticker_rotation_interval_s = float(
        config.display_news_ticker_rotation_interval_s
    )
    if not math.isfinite(normalized_display_news_ticker_rotation_interval_s):
        raise ValueError("display_news_ticker_rotation_interval_s must be finite")
    normalized_display_news_ticker_rotation_interval_s = max(
        4.0, normalized_display_news_ticker_rotation_interval_s
    )
    normalized_display_news_ticker_timeout_s = float(
        config.display_news_ticker_timeout_s
    )
    if not math.isfinite(normalized_display_news_ticker_timeout_s):
        raise ValueError("display_news_ticker_timeout_s must be finite")
    normalized_display_news_ticker_timeout_s = max(
        0.5, normalized_display_news_ticker_timeout_s
    )
    normalized_display_news_ticker_max_items = max(
        1, int(config.display_news_ticker_max_items)
    )
    normalized_display_face_cue_path = (
        str(
            config.display_face_cue_path or "artifacts/stores/ops/display_face_cue.json"
        ).strip()
        or "artifacts/stores/ops/display_face_cue.json"
    )
    normalized_display_emoji_cue_path = (
        str(
            config.display_emoji_cue_path or "artifacts/stores/ops/display_emoji.json"
        ).strip()
        or "artifacts/stores/ops/display_emoji.json"
    )
    normalized_display_ambient_impulse_path = (
        str(
            config.display_ambient_impulse_path
            or "artifacts/stores/ops/display_ambient_impulse.json"
        ).strip()
        or "artifacts/stores/ops/display_ambient_impulse.json"
    )
    normalized_display_service_connect_path = (
        str(
            config.display_service_connect_path
            or "artifacts/stores/ops/display_service_connect.json"
        ).strip()
        or "artifacts/stores/ops/display_service_connect.json"
    )
    normalized_display_reserve_bus_plan_path = (
        str(
            config.display_reserve_bus_plan_path
            or "artifacts/stores/ops/display_reserve_bus_plan.json"
        ).strip()
        or "artifacts/stores/ops/display_reserve_bus_plan.json"
    )
    normalized_display_reserve_bus_prepared_plan_path = (
        str(
            config.display_reserve_bus_prepared_plan_path
            or "artifacts/stores/ops/display_reserve_bus_plan_prepared.json"
        ).strip()
        or "artifacts/stores/ops/display_reserve_bus_plan_prepared.json"
    )
    normalized_display_reserve_bus_maintenance_state_path = (
        str(
            config.display_reserve_bus_maintenance_state_path
            or "artifacts/stores/ops/display_reserve_bus_maintenance.json"
        ).strip()
        or "artifacts/stores/ops/display_reserve_bus_maintenance.json"
    )
    normalized_display_reserve_bus_refresh_after_local = (
        str(config.display_reserve_bus_refresh_after_local or "05:30").strip()
        or "05:30"
    )
    normalized_display_reserve_bus_nightly_after_local = (
        str(config.display_reserve_bus_nightly_after_local or "00:30").strip()
        or "00:30"
    )
    normalized_display_presentation_path = (
        str(
            config.display_presentation_path
            or "artifacts/stores/ops/display_presentation.json"
        ).strip()
        or "artifacts/stores/ops/display_presentation.json"
    )
    normalized_proactive_quiet_hours_start_local = (
        str(config.proactive_quiet_hours_start_local or "21:00").strip() or "21:00"
    )
    normalized_proactive_quiet_hours_end_local = (
        str(config.proactive_quiet_hours_end_local or "07:00").strip() or "07:00"
    )
    normalized_display_news_ticker_store_path = (
        str(
            config.display_news_ticker_store_path
            or "artifacts/stores/ops/display_news_ticker.json"
        ).strip()
        or "artifacts/stores/ops/display_news_ticker.json"
    )
    normalized_voice_orchestrator_ws_url = str(
        config.voice_orchestrator_ws_url or ""
    ).strip()
    normalized_camera_host_mode = _parse_camera_host_mode(
        config.camera_host_mode, default="onboard"
    )
    normalized_camera_second_pi_base_url = _parse_optional_url(
        None
        if config.camera_second_pi_base_url is None
        else str(config.camera_second_pi_base_url),
        strip_trailing_slash=True,
    )
    normalized_camera_proxy_snapshot_url = _parse_optional_url(
        None
        if config.camera_proxy_snapshot_url is None
        else str(config.camera_proxy_snapshot_url),
    )
    normalized_proactive_remote_camera_base_url = _parse_optional_url(
        None
        if config.proactive_remote_camera_base_url is None
        else str(config.proactive_remote_camera_base_url),
        strip_trailing_slash=True,
    )
    normalized_drone_base_url = _parse_optional_url(
        None if config.drone_base_url is None else str(config.drone_base_url),
        strip_trailing_slash=True,
    )
    normalized_proactive_vision_provider = (
        str(config.proactive_vision_provider or "local_first").strip().lower()
        or "local_first"
    )
    if config.voice_orchestrator_enabled and not normalized_voice_orchestrator_ws_url:
        raise ValueError(
            "voice_orchestrator_enabled requires TWINR_VOICE_ORCHESTRATOR_WS_URL; "
            "Twinr must not fall back to an implicit voice gateway endpoint."
        )
    normalized_values = locals().copy()
    apply_general_updates(config, normalized_values)
    apply_display_updates(config, normalized_values)
    apply_attention_servo_updates(config, normalized_values)
