"""Shared defaults and marker metadata for environment-profile compilation."""

from __future__ import annotations

_DEFAULT_TIMEZONE_NAME = "Europe/Berlin"
_DEFAULT_BASELINE_DAYS = 14
_DEFAULT_HISTORY_DAYS = 42
_DEFAULT_MIN_BASELINE_DAYS = 5
_DEFAULT_LONG_BASELINE_DAYS = 56
_DEFAULT_EPOCH_MINUTES = 5
_DEFAULT_TRANSITION_WINDOW_S = 90.0
_DEFAULT_DAY_START_HOUR = 6
_DEFAULT_NIGHT_START_HOUR = 22
_DEFAULT_IQR_MULTIPLIER = 1.5
_DEFAULT_ACUTE_Z_THRESHOLD = 3.0
_DEFAULT_ACUTE_EMPIRICAL_Q = 0.01
_DEFAULT_DRIFT_MIN_SIGMA = 1.5
_DEFAULT_DRIFT_MIN_DAYS = 5
_DEFAULT_REGIME_ACCEPT_DAYS = 10
_DEFAULT_MIN_COVERAGE_RATIO = 0.8
_MAX_SOURCE_EVENT_IDS = 32
_SMART_HOME_ENVIRONMENT_DOMAIN = "smart_home_environment"
_MOTION_SIGNAL_TYPE = "motion_node_activity"
_HEALTH_SIGNAL_TYPE = "node_health"
_WEEKDAY_CLASSES = ("all_days", "weekday", "weekend")
_BASELINE_KINDS = ("short", "long")
_BASELINE_MARKER_NAMES = (
    "active_epoch_count_day",
    "first_activity_minute_local",
    "last_activity_minute_local",
    "longest_daytime_inactivity_min",
    "night_activity_epoch_count",
    "unique_active_node_count_day",
    "mean_active_node_count_per_active_epoch",
    "node_entropy_day",
    "dominant_node_share_day",
    "transition_count_day",
    "transition_entropy_day",
    "fragmentation_index_day",
    "motion_burst_count_day",
    "inter_event_interval_median_day",
    "inter_event_interval_iqr_day",
    "active_epoch_run_length_median_day",
    "inactive_gap_entropy_day",
    "circadian_similarity_14d",
    "transition_graph_divergence_14d",
    "node_usage_divergence_14d",
    "sensor_coverage_ratio_day",
)
