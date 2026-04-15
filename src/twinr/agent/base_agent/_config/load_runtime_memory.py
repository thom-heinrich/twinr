"""Load runtime-state, long-term-memory, automation, and smart-home settings."""

from __future__ import annotations

from .context import ConfigLoadContext
from .parsing import (
    _parse_bool,
    _parse_csv_strings,
    _parse_float,
    _parse_optional_int,
)


def load_runtime_memory_config(context: ConfigLoadContext) -> dict[str, object]:
    """Return the config fields owned by this loading domain."""

    get_value = context.get_value
    project_root = context.project_root
    default_remote_runtime_check_mode = context.default_remote_runtime_check_mode

    return {
        "web_host": get_value("TWINR_WEB_HOST", "0.0.0.0") or "0.0.0.0",
        "web_port": int(get_value("TWINR_WEB_PORT", "1337") or "1337"),
        "runtime_state_path": get_value(
            "TWINR_RUNTIME_STATE_PATH",
            str(project_root / "state" / "runtime-state.json"),
        )
        or str(project_root / "state" / "runtime-state.json"),
        "memory_markdown_path": get_value(
            "TWINR_MEMORY_MARKDOWN_PATH", str(project_root / "state" / "MEMORY.md")
        )
        or str(project_root / "state" / "MEMORY.md"),
        "reminder_store_path": get_value(
            "TWINR_REMINDER_STORE_PATH", str(project_root / "state" / "reminders.json")
        )
        or str(project_root / "state" / "reminders.json"),
        "automation_store_path": get_value(
            "TWINR_AUTOMATION_STORE_PATH",
            str(project_root / "state" / "automations.json"),
        )
        or str(project_root / "state" / "automations.json"),
        "voice_profile_store_path": get_value(
            "TWINR_VOICE_PROFILE_STORE_PATH",
            str(project_root / "state" / "voice_profile.json"),
        )
        or str(project_root / "state" / "voice_profile.json"),
        "adaptive_timing_enabled": _parse_bool(
            get_value("TWINR_ADAPTIVE_TIMING_ENABLED"), True
        ),
        "adaptive_timing_store_path": get_value(
            "TWINR_ADAPTIVE_TIMING_STORE_PATH",
            str(project_root / "state" / "adaptive_timing.json"),
        )
        or str(project_root / "state" / "adaptive_timing.json"),
        "adaptive_timing_pause_grace_ms": int(
            get_value("TWINR_ADAPTIVE_TIMING_PAUSE_GRACE_MS", "450") or "450"
        ),
        "long_term_memory_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_ENABLED"), False
        ),
        "long_term_memory_backend": get_value(
            "TWINR_LONG_TERM_MEMORY_BACKEND", "chonkydb"
        )
        or "chonkydb",
        "long_term_memory_mode": (
            get_value("TWINR_LONG_TERM_MEMORY_MODE", "local_first") or "local_first"
        )
        .strip()
        .lower(),
        "long_term_memory_remote_required": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED"), False
        ),
        "long_term_memory_remote_required_failure_threshold": _parse_optional_int(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED_FAILURE_THRESHOLD")
        ),
        "long_term_memory_remote_namespace": get_value(
            "TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE"
        )
        or None,
        "long_term_memory_path": get_value(
            "TWINR_LONG_TERM_MEMORY_PATH", str(project_root / "state" / "chonkydb")
        )
        or str(project_root / "state" / "chonkydb"),
        "long_term_memory_background_store_turns": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS"), True
        ),
        "long_term_memory_write_queue_size": int(
            get_value("TWINR_LONG_TERM_MEMORY_WRITE_QUEUE_SIZE", "32") or "32"
        ),
        "long_term_memory_recall_limit": int(
            get_value("TWINR_LONG_TERM_MEMORY_RECALL_LIMIT", "3") or "3"
        ),
        "long_term_memory_fast_topic_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_FAST_TOPIC_ENABLED"), True
        ),
        "long_term_memory_fast_topic_limit": int(
            get_value("TWINR_LONG_TERM_MEMORY_FAST_TOPIC_LIMIT", "3") or "3"
        ),
        "long_term_memory_fast_topic_timeout_s": _parse_float(
            # ChonkyDB's advanced scope-query path keeps ~0.75 s internal headroom,
            # so Twinr's required fast-topic budget must stay above that floor or
            # the effective compute window collapses into deterministic 504s/503s.
            get_value("TWINR_LONG_TERM_MEMORY_FAST_TOPIC_TIMEOUT_S"), 1.2, minimum=0.05
        ),
        "long_term_memory_query_rewrite_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_QUERY_REWRITE_ENABLED"), True
        ),
        "long_term_memory_remote_read_timeout_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_READ_TIMEOUT_S"), 8.0
        ),
        "long_term_memory_remote_write_timeout_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WRITE_TIMEOUT_S"), 15.0
        ),
        "long_term_memory_remote_keepalive_interval_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_KEEPALIVE_INTERVAL_S"),
            5.0,
            minimum=0.1,
        ),
        "long_term_memory_remote_runtime_check_mode": (
            get_value(
                "TWINR_LONG_TERM_MEMORY_REMOTE_RUNTIME_CHECK_MODE",
                default_remote_runtime_check_mode,
            )
            or default_remote_runtime_check_mode
        )
        .strip()
        .lower(),
        "long_term_memory_remote_watchdog_startup_wait_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_STARTUP_WAIT_S"),
            30.0,
            minimum=0.0,
            maximum=300.0,
        ),
        "long_term_memory_remote_watchdog_interval_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_INTERVAL_S"),
            1.0,
            minimum=0.1,
        ),
        "long_term_memory_remote_watchdog_probe_mode": (
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_PROBE_MODE", "auto")
            or "auto"
        )
        .strip()
        .lower(),
        "long_term_memory_remote_watchdog_probe_timeout_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_PROBE_TIMEOUT_S"),
            15.0,
            minimum=1.0,
            maximum=120.0,
        ),
        "long_term_memory_remote_watchdog_startup_probe_timeout_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_STARTUP_PROBE_TIMEOUT_S"),
            180.0,
            minimum=15.0,
            maximum=300.0,
        ),
        "long_term_memory_remote_watchdog_history_limit": max(
            1,
            int(
                get_value(
                    "TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_HISTORY_LIMIT", "3600"
                )
                or "3600"
            ),
        ),
        "long_term_memory_remote_max_content_chars": int(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_MAX_CONTENT_CHARS", "2000000")
            or "2000000"
        ),
        "long_term_memory_remote_shard_max_content_chars": int(
            get_value(
                "TWINR_LONG_TERM_MEMORY_REMOTE_SHARD_MAX_CONTENT_CHARS", "1000000"
            )
            or "1000000"
        ),
        "long_term_memory_remote_retry_attempts": int(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_RETRY_ATTEMPTS", "3") or "3"
        ),
        "long_term_memory_remote_retry_backoff_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_RETRY_BACKOFF_S"), 1.0
        ),
        "long_term_memory_remote_flush_timeout_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_FLUSH_TIMEOUT_S"), 60.0
        ),
        "long_term_memory_remote_read_cache_ttl_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_REMOTE_READ_CACHE_TTL_S"),
            0.0,
            minimum=0.0,
        ),
        "long_term_memory_turn_extractor_model": get_value(
            "TWINR_LONG_TERM_MEMORY_TURN_EXTRACTOR_MODEL"
        )
        or None,
        "long_term_memory_turn_extractor_max_output_tokens": int(
            get_value("TWINR_LONG_TERM_MEMORY_TURN_EXTRACTOR_MAX_OUTPUT_TOKENS", "2200")
            or "2200"
        ),
        "long_term_memory_midterm_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_MIDTERM_ENABLED"), True
        ),
        "long_term_memory_midterm_limit": int(
            get_value("TWINR_LONG_TERM_MEMORY_MIDTERM_LIMIT", "4") or "4"
        ),
        "long_term_memory_reflection_window_size": int(
            get_value("TWINR_LONG_TERM_MEMORY_REFLECTION_WINDOW_SIZE", "18") or "18"
        ),
        "long_term_memory_reflection_compiler_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_ENABLED"), True
        ),
        "long_term_memory_reflection_compiler_model": get_value(
            "TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_MODEL"
        )
        or None,
        "long_term_memory_reflection_compiler_max_output_tokens": int(
            get_value(
                "TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_MAX_OUTPUT_TOKENS", "900"
            )
            or "900"
        ),
        "long_term_memory_subtext_compiler_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_ENABLED"), True
        ),
        "long_term_memory_subtext_compiler_model": get_value(
            "TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_MODEL"
        )
        or None,
        "long_term_memory_subtext_compiler_max_output_tokens": int(
            get_value(
                "TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_MAX_OUTPUT_TOKENS", "520"
            )
            or "520"
        ),
        "long_term_memory_proactive_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_ENABLED"), False
        ),
        "long_term_memory_proactive_poll_interval_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_POLL_INTERVAL_S"), 30.0
        ),
        "long_term_memory_proactive_min_confidence": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_MIN_CONFIDENCE"), 0.72
        ),
        "long_term_memory_proactive_repeat_cooldown_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_REPEAT_COOLDOWN_S"),
            6.0 * 60.0 * 60.0,
        ),
        "long_term_memory_proactive_skip_cooldown_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_SKIP_COOLDOWN_S"), 30.0 * 60.0
        ),
        "long_term_memory_proactive_reservation_ttl_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_RESERVATION_TTL_S"), 90.0
        ),
        "long_term_memory_proactive_allow_sensitive": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_ALLOW_SENSITIVE"), False
        ),
        "long_term_memory_proactive_history_limit": int(
            get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_HISTORY_LIMIT", "128") or "128"
        ),
        "long_term_memory_sensor_memory_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_SENSOR_MEMORY_ENABLED"), False
        ),
        "long_term_memory_sensor_baseline_days": int(
            get_value("TWINR_LONG_TERM_MEMORY_SENSOR_BASELINE_DAYS", "21") or "21"
        ),
        "long_term_memory_sensor_min_days_observed": int(
            get_value("TWINR_LONG_TERM_MEMORY_SENSOR_MIN_DAYS_OBSERVED", "6") or "6"
        ),
        "long_term_memory_sensor_min_routine_ratio": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_SENSOR_MIN_ROUTINE_RATIO"), 0.55
        ),
        "long_term_memory_sensor_deviation_min_delta": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_SENSOR_DEVIATION_MIN_DELTA"), 0.45
        ),
        "long_term_memory_environment_short_baseline_days": int(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_SHORT_BASELINE_DAYS", "14")
            or "14"
        ),
        "long_term_memory_environment_long_baseline_days": int(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_LONG_BASELINE_DAYS", "56")
            or "56"
        ),
        "long_term_memory_environment_min_baseline_days": int(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_MIN_BASELINE_DAYS", "7")
            or "7"
        ),
        "long_term_memory_environment_acute_z_threshold": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_ACUTE_Z_THRESHOLD"), 3.0
        ),
        "long_term_memory_environment_acute_empirical_q": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_ACUTE_EMPIRICAL_Q"), 0.01
        ),
        "long_term_memory_environment_drift_min_sigma": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_DRIFT_MIN_SIGMA"), 1.5
        ),
        "long_term_memory_environment_drift_min_days": int(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_DRIFT_MIN_DAYS", "5") or "5"
        ),
        "long_term_memory_environment_regime_accept_days": int(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_REGIME_ACCEPT_DAYS", "10")
            or "10"
        ),
        "long_term_memory_environment_min_coverage_ratio": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_ENVIRONMENT_MIN_COVERAGE_RATIO"), 0.8
        ),
        "long_term_memory_retention_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_RETENTION_ENABLED"), True
        ),
        "long_term_memory_retention_mode": (
            get_value("TWINR_LONG_TERM_MEMORY_RETENTION_MODE", "conservative")
            or "conservative"
        )
        .strip()
        .lower(),
        "long_term_memory_retention_run_interval_s": _parse_float(
            get_value("TWINR_LONG_TERM_MEMORY_RETENTION_RUN_INTERVAL_S"), 300.0
        ),
        "long_term_memory_archive_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_ARCHIVE_ENABLED"), True
        ),
        "long_term_memory_migration_enabled": _parse_bool(
            get_value("TWINR_LONG_TERM_MEMORY_MIGRATION_ENABLED"), True
        ),
        "long_term_memory_migration_batch_size": int(
            get_value("TWINR_LONG_TERM_MEMORY_MIGRATION_BATCH_SIZE", "64") or "64"
        ),
        "long_term_memory_remote_bulk_request_max_bytes": int(
            get_value(
                "TWINR_LONG_TERM_MEMORY_REMOTE_BULK_REQUEST_MAX_BYTES", str(512 * 1024)
            )
            or str(512 * 1024)
        ),
        "chonkydb_base_url": get_value("TWINR_CHONKYDB_BASE_URL")
        or get_value("CCODEX_MEMORY_BASE_URL"),
        "chonkydb_api_key": get_value("TWINR_CHONKYDB_API_KEY")
        or get_value("CCODEX_MEMORY_API_KEY"),
        "chonkydb_api_key_header": get_value(
            "TWINR_CHONKYDB_API_KEY_HEADER", "x-api-key"
        )
        or "x-api-key",
        "chonkydb_allow_bearer_auth": _parse_bool(
            get_value("TWINR_CHONKYDB_ALLOW_BEARER_AUTH"), False
        ),
        "chonkydb_timeout_s": _parse_float(get_value("TWINR_CHONKYDB_TIMEOUT_S"), 20.0),
        "chonkydb_max_response_bytes": int(
            get_value("TWINR_CHONKYDB_MAX_RESPONSE_BYTES", str(64 * 1024 * 1024))
            or str(64 * 1024 * 1024)
        ),
        "restore_runtime_state_on_startup": _parse_bool(
            get_value("TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP"), False
        ),
        "reminder_poll_interval_s": _parse_float(
            get_value("TWINR_REMINDER_POLL_INTERVAL_S"), 1.0
        ),
        "reminder_retry_delay_s": _parse_float(
            get_value("TWINR_REMINDER_RETRY_DELAY_S"), 90.0
        ),
        "reminder_max_entries": int(
            get_value("TWINR_REMINDER_MAX_ENTRIES", "48") or "48"
        ),
        "automation_poll_interval_s": _parse_float(
            get_value("TWINR_AUTOMATION_POLL_INTERVAL_S"), 5.0
        ),
        "automation_max_entries": int(
            get_value("TWINR_AUTOMATION_MAX_ENTRIES", "96") or "96"
        ),
        "nightly_orchestration_enabled": _parse_bool(
            get_value("TWINR_NIGHTLY_ORCHESTRATION_ENABLED"), True
        ),
        "nightly_orchestration_after_local": get_value(
            "TWINR_NIGHTLY_ORCHESTRATION_AFTER_LOCAL",
            "00:30",
        )
        or "00:30",
        "nightly_orchestration_poll_interval_s": _parse_float(
            get_value("TWINR_NIGHTLY_ORCHESTRATION_POLL_INTERVAL_S"),
            300.0,
            minimum=30.0,
        ),
        "nightly_orchestration_flush_timeout_s": _parse_float(
            get_value("TWINR_NIGHTLY_ORCHESTRATION_FLUSH_TIMEOUT_S"),
            15.0,
            minimum=1.0,
            maximum=300.0,
        ),
        "nightly_orchestration_state_path": get_value(
            "TWINR_NIGHTLY_ORCHESTRATION_STATE_PATH",
            "artifacts/stores/ops/nightly_run_state.json",
        )
        or "artifacts/stores/ops/nightly_run_state.json",
        "nightly_prepared_digest_path": get_value(
            "TWINR_NIGHTLY_PREPARED_DIGEST_PATH",
            "artifacts/stores/ops/nightly_prepared_digest.json",
        )
        or "artifacts/stores/ops/nightly_prepared_digest.json",
        "nightly_consolidation_summary_path": get_value(
            "TWINR_NIGHTLY_CONSOLIDATION_SUMMARY_PATH",
            "artifacts/stores/ops/nightly_consolidation_summary.json",
        )
        or "artifacts/stores/ops/nightly_consolidation_summary.json",
        "nightly_digest_reminder_limit": max(
            1,
            int(get_value("TWINR_NIGHTLY_DIGEST_REMINDER_LIMIT", "6") or "6"),
        ),
        "nightly_digest_headline_limit": max(
            1,
            int(get_value("TWINR_NIGHTLY_DIGEST_HEADLINE_LIMIT", "5") or "5"),
        ),
        "nightly_live_web_augmentation_enabled": _parse_bool(
            get_value("TWINR_NIGHTLY_LIVE_WEB_AUGMENTATION_ENABLED"),
            True,
        ),
        "nightly_live_web_query_limit": max(
            0,
            int(get_value("TWINR_NIGHTLY_LIVE_WEB_QUERY_LIMIT", "2") or "2"),
        ),
        "browser_automation_enabled": _parse_bool(
            get_value("TWINR_BROWSER_AUTOMATION_ENABLED"), False
        ),
        "browser_automation_workspace_path": get_value(
            "TWINR_BROWSER_AUTOMATION_WORKSPACE_PATH", "browser_automation"
        )
        or "browser_automation",
        "browser_automation_entry_module": get_value(
            "TWINR_BROWSER_AUTOMATION_ENTRY_MODULE", "adapter.py"
        )
        or "adapter.py",
        "smart_home_background_worker_enabled": _parse_bool(
            get_value("TWINR_SMART_HOME_BACKGROUND_WORKER_ENABLED"), True
        ),
        "smart_home_background_idle_sleep_s": _parse_float(
            get_value("TWINR_SMART_HOME_BACKGROUND_IDLE_SLEEP_S"), 1.0, minimum=0.1
        ),
        "smart_home_background_retry_delay_s": _parse_float(
            get_value("TWINR_SMART_HOME_BACKGROUND_RETRY_DELAY_S"), 2.0, minimum=0.1
        ),
        "smart_home_background_batch_limit": max(
            1, int(get_value("TWINR_SMART_HOME_BACKGROUND_BATCH_LIMIT", "8") or "8")
        ),
        "smart_home_same_room_entity_ids": _parse_csv_strings(
            get_value("TWINR_SMART_HOME_SAME_ROOM_ENTITY_IDS"), ()
        ),
        "smart_home_same_room_motion_window_s": _parse_float(
            get_value("TWINR_SMART_HOME_SAME_ROOM_MOTION_WINDOW_S"), 90.0, minimum=0.0
        ),
        "smart_home_same_room_button_window_s": _parse_float(
            get_value("TWINR_SMART_HOME_SAME_ROOM_BUTTON_WINDOW_S"), 30.0, minimum=0.0
        ),
        "smart_home_home_occupancy_window_s": _parse_float(
            get_value("TWINR_SMART_HOME_HOME_OCCUPANCY_WINDOW_S"), 300.0, minimum=0.0
        ),
        "smart_home_stream_stale_after_s": _parse_float(
            get_value("TWINR_SMART_HOME_STREAM_STALE_AFTER_S"), 120.0, minimum=0.0
        ),
        "voice_profile_min_sample_ms": int(
            get_value("TWINR_VOICE_PROFILE_MIN_SAMPLE_MS", "1200") or "1200"
        ),
        "voice_profile_likely_threshold": _parse_float(
            get_value("TWINR_VOICE_PROFILE_LIKELY_THRESHOLD"), 0.72
        ),
        "voice_profile_uncertain_threshold": _parse_float(
            get_value("TWINR_VOICE_PROFILE_UNCERTAIN_THRESHOLD"), 0.55
        ),
        "voice_profile_max_samples": int(
            get_value("TWINR_VOICE_PROFILE_MAX_SAMPLES", "6") or "6"
        ),
        "voice_familiar_speaker_min_confidence": _parse_float(
            get_value("TWINR_VOICE_FAMILIAR_SPEAKER_MIN_CONFIDENCE"), 0.82
        ),
        "voice_profile_passive_update_enabled": _parse_bool(
            get_value("TWINR_VOICE_PROFILE_PASSIVE_UPDATE_ENABLED"), True
        ),
        "voice_profile_passive_update_min_confidence": _parse_float(
            get_value("TWINR_VOICE_PROFILE_PASSIVE_UPDATE_MIN_CONFIDENCE"), 0.86
        ),
        "voice_profile_passive_update_min_duration_ms": int(
            get_value("TWINR_VOICE_PROFILE_PASSIVE_UPDATE_MIN_DURATION_MS", "2500")
            or "2500"
        ),
        "speech_pause_ms": int(get_value("TWINR_SPEECH_PAUSE_MS", "1200") or "1200"),
        "memory_max_turns": int(get_value("TWINR_MEMORY_MAX_TURNS", "20") or "20"),
        "memory_keep_recent": int(get_value("TWINR_MEMORY_KEEP_RECENT", "10") or "10"),
    }
