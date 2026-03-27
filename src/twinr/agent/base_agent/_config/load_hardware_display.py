"""Load GPIO, display, printer, and attention-servo hardware settings."""

from __future__ import annotations

from .context import ConfigLoadContext
from .constants import (
    DEFAULT_BUTTON_PROBE_LINES,
)
from .parsing import (
    _default_display_poll_interval_s,
    _parse_attention_servo_control_mode,
    _parse_attention_servo_driver,
    _parse_bool,
    _parse_csv_ints,
    _parse_csv_strings,
    _parse_float,
    _parse_optional_bool,
    _parse_optional_int,
)


def load_hardware_display_config(context: ConfigLoadContext) -> dict[str, object]:
    """Return the config fields owned by this loading domain."""

    get_value = context.get_value
    project_root = context.project_root

    return {
        "gpio_chip": get_value("TWINR_GPIO_CHIP", "gpiochip0") or "gpiochip0",
        "green_button_gpio": _parse_optional_int(get_value("TWINR_GREEN_BUTTON_GPIO")),
        "yellow_button_gpio": _parse_optional_int(
            get_value("TWINR_YELLOW_BUTTON_GPIO")
        ),
        "pir_motion_gpio": _parse_optional_int(get_value("TWINR_PIR_MOTION_GPIO")),
        "pir_active_high": _parse_bool(get_value("TWINR_PIR_ACTIVE_HIGH"), True),
        "pir_bias": (get_value("TWINR_PIR_BIAS", "pull-down") or "pull-down")
        .strip()
        .lower(),
        "pir_debounce_ms": int(get_value("TWINR_PIR_DEBOUNCE_MS", "120") or "120"),
        "button_active_low": _parse_bool(get_value("TWINR_BUTTON_ACTIVE_LOW"), True),
        "button_bias": (get_value("TWINR_BUTTON_BIAS", "pull-up") or "pull-up")
        .strip()
        .lower(),
        "button_debounce_ms": int(get_value("TWINR_BUTTON_DEBOUNCE_MS", "80") or "80"),
        "button_probe_lines": _parse_csv_ints(
            get_value("TWINR_BUTTON_PROBE_LINES"), DEFAULT_BUTTON_PROBE_LINES
        ),
        "display_driver": get_value("TWINR_DISPLAY_DRIVER", "hdmi_fbdev")
        or "hdmi_fbdev",
        "display_companion_enabled": _parse_optional_bool(
            get_value("TWINR_DISPLAY_COMPANION_ENABLED")
        ),
        "respeaker_led_enabled": _parse_optional_bool(
            get_value("TWINR_RESPEAKER_LED_ENABLED")
        ),
        "display_fb_path": get_value("TWINR_DISPLAY_FB_PATH", "/dev/fb0") or "/dev/fb0",
        "display_wayland_display": get_value(
            "TWINR_DISPLAY_WAYLAND_DISPLAY", "wayland-0"
        )
        or "wayland-0",
        "display_wayland_runtime_dir": get_value("TWINR_DISPLAY_WAYLAND_RUNTIME_DIR"),
        "display_face_cue_path": get_value(
            "TWINR_DISPLAY_FACE_CUE_PATH", "artifacts/stores/ops/display_face_cue.json"
        )
        or "artifacts/stores/ops/display_face_cue.json",
        "display_face_cue_ttl_s": _parse_float(
            get_value("TWINR_DISPLAY_FACE_CUE_TTL_S"), 4.0, minimum=0.1
        ),
        "display_emoji_cue_path": get_value(
            "TWINR_DISPLAY_EMOJI_CUE_PATH", "artifacts/stores/ops/display_emoji.json"
        )
        or "artifacts/stores/ops/display_emoji.json",
        "display_emoji_cue_ttl_s": _parse_float(
            get_value("TWINR_DISPLAY_EMOJI_CUE_TTL_S"), 6.0, minimum=0.1
        ),
        "display_ambient_impulse_path": get_value(
            "TWINR_DISPLAY_AMBIENT_IMPULSE_PATH",
            "artifacts/stores/ops/display_ambient_impulse.json",
        )
        or "artifacts/stores/ops/display_ambient_impulse.json",
        "display_ambient_impulse_ttl_s": _parse_float(
            get_value("TWINR_DISPLAY_AMBIENT_IMPULSE_TTL_S"), 18.0, minimum=0.1
        ),
        "display_service_connect_path": get_value(
            "TWINR_DISPLAY_SERVICE_CONNECT_PATH",
            "artifacts/stores/ops/display_service_connect.json",
        )
        or "artifacts/stores/ops/display_service_connect.json",
        "display_ambient_impulses_enabled": _parse_bool(
            get_value("TWINR_DISPLAY_AMBIENT_IMPULSES_ENABLED"), True
        ),
        "display_reserve_generation_enabled": _parse_bool(
            get_value("TWINR_DISPLAY_RESERVE_GENERATION_ENABLED"), True
        ),
        "display_reserve_generation_model": get_value(
            "TWINR_DISPLAY_RESERVE_GENERATION_MODEL", ""
        )
        or "",
        "display_reserve_generation_reasoning_effort": get_value(
            "TWINR_DISPLAY_RESERVE_GENERATION_REASONING_EFFORT", "low"
        )
        or "low",
        "display_reserve_generation_timeout_seconds": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_GENERATION_TIMEOUT_SECONDS"),
            20.0,
            minimum=1.0,
        ),
        "display_reserve_generation_max_output_tokens": max(
            128,
            int(
                get_value("TWINR_DISPLAY_RESERVE_GENERATION_MAX_OUTPUT_TOKENS", "900")
                or "900"
            ),
        ),
        "display_reserve_generation_batch_size": max(
            1, int(get_value("TWINR_DISPLAY_RESERVE_GENERATION_BATCH_SIZE", "2") or "2")
        ),
        "display_reserve_generation_variants_per_candidate": max(
            1,
            int(
                get_value(
                    "TWINR_DISPLAY_RESERVE_GENERATION_VARIANTS_PER_CANDIDATE", "3"
                )
                or "3"
            ),
        ),
        "display_reserve_bus_plan_path": get_value(
            "TWINR_DISPLAY_RESERVE_BUS_PLAN_PATH",
            "artifacts/stores/ops/display_reserve_bus_plan.json",
        )
        or "artifacts/stores/ops/display_reserve_bus_plan.json",
        "display_reserve_bus_prepared_plan_path": get_value(
            "TWINR_DISPLAY_RESERVE_BUS_PREPARED_PLAN_PATH",
            "artifacts/stores/ops/display_reserve_bus_plan_prepared.json",
        )
        or "artifacts/stores/ops/display_reserve_bus_plan_prepared.json",
        "display_reserve_bus_maintenance_state_path": get_value(
            "TWINR_DISPLAY_RESERVE_BUS_MAINTENANCE_STATE_PATH",
            "artifacts/stores/ops/display_reserve_bus_maintenance.json",
        )
        or "artifacts/stores/ops/display_reserve_bus_maintenance.json",
        "display_reserve_bus_refresh_after_local": get_value(
            "TWINR_DISPLAY_RESERVE_BUS_REFRESH_AFTER_LOCAL", "05:30"
        )
        or "05:30",
        "display_reserve_bus_nightly_enabled": _parse_bool(
            get_value("TWINR_DISPLAY_RESERVE_BUS_NIGHTLY_ENABLED"), True
        ),
        "display_reserve_bus_nightly_after_local": get_value(
            "TWINR_DISPLAY_RESERVE_BUS_NIGHTLY_AFTER_LOCAL", "00:30"
        )
        or "00:30",
        "display_reserve_bus_nightly_poll_interval_s": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_NIGHTLY_POLL_INTERVAL_S"),
            300.0,
            minimum=30.0,
        ),
        "display_reserve_bus_candidate_limit": int(
            get_value("TWINR_DISPLAY_RESERVE_BUS_CANDIDATE_LIMIT", "20") or "20"
        ),
        "display_reserve_bus_items_per_day": int(
            get_value("TWINR_DISPLAY_RESERVE_BUS_ITEMS_PER_DAY", "20") or "20"
        ),
        "display_reserve_bus_topic_gap": int(
            get_value("TWINR_DISPLAY_RESERVE_BUS_TOPIC_GAP", "2") or "2"
        ),
        "display_reserve_bus_learning_window_days": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_LEARNING_WINDOW_DAYS"),
            21.0,
            minimum=3.0,
        ),
        "display_reserve_bus_learning_half_life_days": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_LEARNING_HALF_LIFE_DAYS"),
            7.0,
            minimum=1.0,
        ),
        "display_reserve_bus_reflection_candidate_limit": int(
            get_value("TWINR_DISPLAY_RESERVE_BUS_REFLECTION_CANDIDATE_LIMIT", "3")
            or "3"
        ),
        "display_reserve_bus_reflection_max_age_days": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_REFLECTION_MAX_AGE_DAYS"),
            14.0,
            minimum=1.0,
        ),
        "display_reserve_bus_min_hold_s": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_MIN_HOLD_S"), 240.0, minimum=60.0
        ),
        "display_reserve_bus_base_hold_s": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_BASE_HOLD_S"), 480.0, minimum=60.0
        ),
        "display_reserve_bus_max_hold_s": _parse_float(
            get_value("TWINR_DISPLAY_RESERVE_BUS_MAX_HOLD_S"), 720.0, minimum=60.0
        ),
        "display_attention_refresh_interval_s": _parse_float(
            get_value("TWINR_DISPLAY_ATTENTION_REFRESH_INTERVAL_S"), 0.2, minimum=0.0
        ),
        "display_attention_session_focus_hold_s": _parse_float(
            get_value("TWINR_DISPLAY_ATTENTION_SESSION_FOCUS_HOLD_S"), 4.5, minimum=0.5
        ),
        "attention_servo_enabled": _parse_bool(
            get_value("TWINR_ATTENTION_SERVO_ENABLED"), False
        ),
        "attention_servo_forensic_trace_enabled": _parse_bool(
            get_value("TWINR_ATTENTION_SERVO_FORENSIC_TRACE_ENABLED"), False
        ),
        "attention_servo_driver": _parse_attention_servo_driver(
            get_value("TWINR_ATTENTION_SERVO_DRIVER"), "auto"
        ),
        "attention_servo_control_mode": _parse_attention_servo_control_mode(
            get_value("TWINR_ATTENTION_SERVO_CONTROL_MODE"), "position"
        ),
        "attention_servo_maestro_device": get_value(
            "TWINR_ATTENTION_SERVO_MAESTRO_DEVICE"
        )
        or None,
        "attention_servo_maestro_channel": _parse_optional_int(
            get_value("TWINR_ATTENTION_SERVO_MAESTRO_CHANNEL")
        ),
        "attention_servo_peer_base_url": get_value(
            "TWINR_ATTENTION_SERVO_PEER_BASE_URL"
        )
        or None,
        "attention_servo_peer_timeout_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_PEER_TIMEOUT_S"), 1.5, minimum=0.1
        ),
        "attention_servo_state_path": get_value(
            "TWINR_ATTENTION_SERVO_STATE_PATH",
            str(project_root / "state" / "attention_servo_state.json"),
        )
        or str(project_root / "state" / "attention_servo_state.json"),
        "attention_servo_estimated_zero_max_uncertainty_degrees": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MAX_UNCERTAINTY_DEGREES"),
            15.0,
            minimum=0.0,
            maximum=180.0,
        ),
        "attention_servo_estimated_zero_settle_tolerance_degrees": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_SETTLE_TOLERANCE_DEGREES"),
            1.0,
            minimum=0.0,
            maximum=180.0,
        ),
        "attention_servo_estimated_zero_speed_scale": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_SPEED_SCALE"),
            0.5,
            minimum=0.0,
            maximum=1.0,
        ),
        "attention_servo_estimated_zero_move_pulse_delta_us": int(
            _parse_float(
                get_value("TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_PULSE_DELTA_US"),
                70,
                minimum=0,
                maximum=500,
            )
        ),
        "attention_servo_estimated_zero_move_period_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_PERIOD_S"),
            0.8,
            minimum=0.05,
            maximum=10.0,
        ),
        "attention_servo_estimated_zero_move_duty_cycle": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_ESTIMATED_ZERO_MOVE_DUTY_CYCLE"),
            0.2,
            minimum=0.05,
            maximum=1.0,
        ),
        "attention_servo_continuous_return_to_zero_after_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_CONTINUOUS_RETURN_TO_ZERO_AFTER_S"),
            0.0,
            minimum=0.0,
            maximum=3600.0,
        ),
        "attention_servo_gpio": _parse_optional_int(
            get_value("TWINR_ATTENTION_SERVO_GPIO")
        ),
        "attention_servo_invert_direction": _parse_bool(
            get_value("TWINR_ATTENTION_SERVO_INVERT_DIRECTION"), False
        ),
        "attention_servo_target_hold_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_TARGET_HOLD_S"), 1.1, minimum=0.0
        ),
        "attention_servo_loss_extrapolation_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_LOSS_EXTRAPOLATION_S"), 0.8, minimum=0.0
        ),
        "attention_servo_loss_extrapolation_gain": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_LOSS_EXTRAPOLATION_GAIN"),
            0.65,
            minimum=0.0,
        ),
        "attention_servo_min_confidence": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_MIN_CONFIDENCE"), 0.58
        ),
        "attention_servo_hold_min_confidence": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_HOLD_MIN_CONFIDENCE"), 0.58
        ),
        "attention_servo_deadband": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_DEADBAND"), 0.045, minimum=0.0
        ),
        "attention_servo_min_pulse_width_us": int(
            get_value("TWINR_ATTENTION_SERVO_MIN_PULSE_WIDTH_US", "1050") or "1050"
        ),
        "attention_servo_center_pulse_width_us": int(
            get_value("TWINR_ATTENTION_SERVO_CENTER_PULSE_WIDTH_US", "1500") or "1500"
        ),
        "attention_servo_max_pulse_width_us": int(
            get_value("TWINR_ATTENTION_SERVO_MAX_PULSE_WIDTH_US", "1950") or "1950"
        ),
        "attention_servo_max_step_us": int(
            get_value("TWINR_ATTENTION_SERVO_MAX_STEP_US", "45") or "45"
        ),
        "attention_servo_target_smoothing_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_TARGET_SMOOTHING_S"), 0.9, minimum=0.0
        ),
        "attention_servo_max_velocity_us_per_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_MAX_VELOCITY_US_PER_S"), 80.0, minimum=1.0
        ),
        "attention_servo_max_acceleration_us_per_s2": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_MAX_ACCELERATION_US_PER_S2"),
            220.0,
            minimum=1.0,
        ),
        "attention_servo_max_jerk_us_per_s3": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_MAX_JERK_US_PER_S3"), 900.0, minimum=1.0
        ),
        "attention_servo_rest_max_velocity_us_per_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_REST_MAX_VELOCITY_US_PER_S"),
            35.0,
            minimum=1.0,
        ),
        "attention_servo_rest_max_acceleration_us_per_s2": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_REST_MAX_ACCELERATION_US_PER_S2"),
            120.0,
            minimum=1.0,
        ),
        "attention_servo_rest_max_jerk_us_per_s3": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_REST_MAX_JERK_US_PER_S3"),
            450.0,
            minimum=1.0,
        ),
        "attention_servo_min_command_delta_us": int(
            get_value("TWINR_ATTENTION_SERVO_MIN_COMMAND_DELTA_US", "8") or "8"
        ),
        "attention_servo_visible_retarget_tolerance_us": int(
            get_value("TWINR_ATTENTION_SERVO_VISIBLE_RETARGET_TOLERANCE_US", "40")
            or "40"
        ),
        "attention_servo_soft_limit_margin_us": int(
            get_value("TWINR_ATTENTION_SERVO_SOFT_LIMIT_MARGIN_US", "70") or "70"
        ),
        "attention_servo_idle_release_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_IDLE_RELEASE_S"), 1.0, minimum=0.0
        ),
        "attention_servo_settled_release_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_SETTLED_RELEASE_S"), 0.0, minimum=0.0
        ),
        "attention_servo_follow_exit_only": _parse_bool(
            get_value("TWINR_ATTENTION_SERVO_FOLLOW_EXIT_ONLY"), False
        ),
        "attention_servo_visible_recenter_interval_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_VISIBLE_RECENTER_INTERVAL_S"),
            30.0,
            minimum=0.0,
        ),
        "attention_servo_visible_recenter_center_tolerance": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_VISIBLE_RECENTER_CENTER_TOLERANCE"),
            0.12,
            minimum=0.0,
        ),
        "attention_servo_mechanical_range_degrees": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_MECHANICAL_RANGE_DEGREES"),
            270.0,
            minimum=30.0,
        ),
        "attention_servo_exit_follow_max_degrees": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_FOLLOW_MAX_DEGREES"),
            60.0,
            minimum=0.0,
        ),
        "attention_servo_exit_activation_delay_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_ACTIVATION_DELAY_S"),
            0.75,
            minimum=0.0,
        ),
        "attention_servo_exit_settle_hold_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_SETTLE_HOLD_S"), 0.6, minimum=0.0
        ),
        "attention_servo_exit_reacquire_center_tolerance": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_REACQUIRE_CENTER_TOLERANCE"),
            0.08,
            minimum=0.0,
        ),
        "attention_servo_exit_visible_edge_threshold": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_VISIBLE_EDGE_THRESHOLD"),
            0.62,
            minimum=0.55,
            maximum=0.95,
        ),
        "attention_servo_exit_visible_box_edge_threshold": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_VISIBLE_BOX_EDGE_THRESHOLD"),
            0.92,
            minimum=0.75,
            maximum=0.99,
        ),
        "attention_servo_exit_cooldown_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_EXIT_COOLDOWN_S"), 30.0, minimum=0.0
        ),
        "attention_servo_continuous_max_speed_degrees_per_s": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_CONTINUOUS_MAX_SPEED_DEGREES_PER_S"),
            120.0,
            minimum=1.0,
        ),
        "attention_servo_continuous_slow_zone_degrees": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_CONTINUOUS_SLOW_ZONE_DEGREES"),
            45.0,
            minimum=0.5,
        ),
        "attention_servo_continuous_stop_tolerance_degrees": _parse_float(
            get_value("TWINR_ATTENTION_SERVO_CONTINUOUS_STOP_TOLERANCE_DEGREES"),
            4.0,
            minimum=0.0,
        ),
        "attention_servo_continuous_min_speed_pulse_delta_us": int(
            get_value("TWINR_ATTENTION_SERVO_CONTINUOUS_MIN_SPEED_PULSE_DELTA_US", "70")
            or "70"
        ),
        "attention_servo_continuous_max_speed_pulse_delta_us": int(
            get_value(
                "TWINR_ATTENTION_SERVO_CONTINUOUS_MAX_SPEED_PULSE_DELTA_US", "160"
            )
            or "160"
        ),
        "display_presentation_path": get_value(
            "TWINR_DISPLAY_PRESENTATION_PATH",
            "artifacts/stores/ops/display_presentation.json",
        )
        or "artifacts/stores/ops/display_presentation.json",
        "display_presentation_ttl_s": _parse_float(
            get_value("TWINR_DISPLAY_PRESENTATION_TTL_S"), 20.0, minimum=0.1
        ),
        "display_vendor_dir": get_value(
            "TWINR_DISPLAY_VENDOR_DIR", "state/display/vendor"
        )
        or "state/display/vendor",
        "display_spi_bus": int(get_value("TWINR_DISPLAY_SPI_BUS", "0") or "0"),
        "display_spi_device": int(get_value("TWINR_DISPLAY_SPI_DEVICE", "0") or "0"),
        "display_cs_gpio": int(get_value("TWINR_DISPLAY_CS_GPIO", "8") or "8"),
        "display_dc_gpio": int(get_value("TWINR_DISPLAY_DC_GPIO", "25") or "25"),
        "display_reset_gpio": int(get_value("TWINR_DISPLAY_RESET_GPIO", "17") or "17"),
        "display_busy_gpio": int(get_value("TWINR_DISPLAY_BUSY_GPIO", "24") or "24"),
        "display_width": int(get_value("TWINR_DISPLAY_WIDTH", "400") or "400"),
        "display_height": int(get_value("TWINR_DISPLAY_HEIGHT", "300") or "300"),
        "display_rotation_degrees": int(
            get_value("TWINR_DISPLAY_ROTATION_DEGREES", "270") or "270"
        ),
        "display_full_refresh_interval": int(
            get_value("TWINR_DISPLAY_FULL_REFRESH_INTERVAL", "0") or "0"
        ),
        "display_busy_timeout_s": _parse_float(
            get_value("TWINR_DISPLAY_BUSY_TIMEOUT_S"), 20.0, minimum=0.1
        ),
        "display_runtime_trace_enabled": _parse_bool(
            get_value("TWINR_DISPLAY_RUNTIME_TRACE_ENABLED"), False
        ),
        "display_poll_interval_s": _parse_float(
            get_value("TWINR_DISPLAY_POLL_INTERVAL_S"),
            _default_display_poll_interval_s(
                get_value("TWINR_DISPLAY_DRIVER", "hdmi_fbdev") or "hdmi_fbdev"
            ),
        ),
        "display_layout": get_value("TWINR_DISPLAY_LAYOUT", "default") or "default",
        "display_news_ticker_enabled": _parse_bool(
            get_value("TWINR_DISPLAY_NEWS_TICKER_ENABLED"), False
        ),
        "display_news_ticker_legacy_feed_urls": _parse_csv_strings(
            get_value("TWINR_DISPLAY_NEWS_TICKER_FEED_URLS"), ()
        ),
        "display_news_ticker_store_path": get_value(
            "TWINR_DISPLAY_NEWS_TICKER_STORE_PATH",
            "artifacts/stores/ops/display_news_ticker.json",
        )
        or "artifacts/stores/ops/display_news_ticker.json",
        "display_news_ticker_refresh_interval_s": _parse_float(
            get_value("TWINR_DISPLAY_NEWS_TICKER_REFRESH_INTERVAL_S"),
            900.0,
            minimum=30.0,
        ),
        "display_news_ticker_rotation_interval_s": _parse_float(
            get_value("TWINR_DISPLAY_NEWS_TICKER_ROTATION_INTERVAL_S"),
            12.0,
            minimum=4.0,
        ),
        "display_news_ticker_max_items": int(
            get_value("TWINR_DISPLAY_NEWS_TICKER_MAX_ITEMS", "12") or "12"
        ),
        "display_news_ticker_timeout_s": _parse_float(
            get_value("TWINR_DISPLAY_NEWS_TICKER_TIMEOUT_S"), 4.0, minimum=0.5
        ),
        "printer_queue": get_value("TWINR_PRINTER_QUEUE", "Thermal_GP58")
        or "Thermal_GP58",
        "printer_device_uri": get_value("TWINR_PRINTER_DEVICE_URI"),
        "printer_header_text": get_value("TWINR_PRINTER_HEADER_TEXT", "TWINR.com")
        or "TWINR.com",
        "printer_feed_lines": int(get_value("TWINR_PRINTER_FEED_LINES", "3") or "3"),
        "printer_line_width": int(get_value("TWINR_PRINTER_LINE_WIDTH", "30") or "30"),
        "print_button_cooldown_s": _parse_float(
            get_value("TWINR_PRINT_BUTTON_COOLDOWN_S"), 2.0
        ),
        "print_max_lines": int(get_value("TWINR_PRINT_MAX_LINES", "8") or "8"),
        "print_max_chars": int(get_value("TWINR_PRINT_MAX_CHARS", "320") or "320"),
        "print_context_turns": int(get_value("TWINR_PRINT_CONTEXT_TURNS", "6") or "6"),
    }
