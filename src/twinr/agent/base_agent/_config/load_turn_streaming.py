"""Load turn control, streaming, audio, and search tuning settings."""

from __future__ import annotations

from .context import ConfigLoadContext
from .parsing import (
    _parse_bool,
    _parse_clamped_float,
    _parse_float,
    _parse_local_semantic_router_mode,
    _parse_optional_text,
)


def load_turn_streaming_config(context: ConfigLoadContext) -> dict[str, object]:
    """Return the config fields owned by this loading domain."""

    get_value = context.get_value

    return {
        "turn_controller_enabled": _parse_bool(
            get_value("TWINR_TURN_CONTROLLER_ENABLED"), True
        ),
        "turn_controller_context_turns": int(
            get_value("TWINR_TURN_CONTROLLER_CONTEXT_TURNS", "4") or "4"
        ),
        "turn_controller_instructions_file": get_value(
            "TWINR_TURN_CONTROLLER_INSTRUCTIONS_FILE", "TURN_CONTROLLER.md"
        )
        or "TURN_CONTROLLER.md",
        "turn_controller_fast_endpoint_enabled": _parse_bool(
            get_value("TWINR_TURN_CONTROLLER_FAST_ENDPOINT_ENABLED"), True
        ),
        "turn_controller_fast_endpoint_min_chars": int(
            get_value("TWINR_TURN_CONTROLLER_FAST_ENDPOINT_MIN_CHARS", "10") or "10"
        ),
        "turn_controller_fast_endpoint_min_confidence": _parse_clamped_float(
            get_value("TWINR_TURN_CONTROLLER_FAST_ENDPOINT_MIN_CONFIDENCE"),
            0.9,
            minimum=0.0,
            maximum=1.0,
        ),
        "turn_controller_backchannel_max_chars": int(
            get_value("TWINR_TURN_CONTROLLER_BACKCHANNEL_MAX_CHARS", "24") or "24"
        ),
        "turn_controller_interrupt_enabled": _parse_bool(
            get_value("TWINR_TURN_CONTROLLER_INTERRUPT_ENABLED"), True
        ),
        "turn_controller_interrupt_window_ms": int(
            get_value("TWINR_TURN_CONTROLLER_INTERRUPT_WINDOW_MS", "420") or "420"
        ),
        "turn_controller_interrupt_poll_ms": int(
            get_value("TWINR_TURN_CONTROLLER_INTERRUPT_POLL_MS", "120") or "120"
        ),
        "turn_controller_interrupt_min_active_ratio": _parse_clamped_float(
            get_value("TWINR_TURN_CONTROLLER_INTERRUPT_MIN_ACTIVE_RATIO"),
            0.18,
            minimum=0.0,
            maximum=1.0,
        ),
        "turn_controller_interrupt_min_transcript_chars": int(
            get_value("TWINR_TURN_CONTROLLER_INTERRUPT_MIN_TRANSCRIPT_CHARS", "4")
            or "4"
        ),
        "turn_controller_interrupt_consecutive_windows": int(
            get_value("TWINR_TURN_CONTROLLER_INTERRUPT_CONSECUTIVE_WINDOWS", "2") or "2"
        ),
        "streaming_early_transcript_enabled": _parse_bool(
            get_value("TWINR_STREAMING_EARLY_TRANSCRIPT_ENABLED"), True
        ),
        "streaming_early_transcript_min_chars": int(
            get_value("TWINR_STREAMING_EARLY_TRANSCRIPT_MIN_CHARS", "10") or "10"
        ),
        "streaming_early_transcript_wait_ms": int(
            get_value("TWINR_STREAMING_EARLY_TRANSCRIPT_WAIT_MS", "250") or "250"
        ),
        "streaming_transcript_verifier_enabled": _parse_bool(
            get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_ENABLED"), True
        ),
        "streaming_transcript_verifier_model": get_value(
            "TWINR_STREAMING_TRANSCRIPT_VERIFIER_MODEL", "gpt-4o-mini-transcribe"
        )
        or "gpt-4o-mini-transcribe",
        "streaming_transcript_verifier_max_words": max(
            1,
            int(get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_WORDS", "6") or "6"),
        ),
        "streaming_transcript_verifier_max_chars": max(
            8,
            int(
                get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_CHARS", "32") or "32"
            ),
        ),
        "streaming_transcript_verifier_min_confidence": _parse_clamped_float(
            get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MIN_CONFIDENCE"),
            0.92,
            minimum=0.0,
            maximum=1.0,
        ),
        "streaming_transcript_verifier_max_capture_ms": max(
            1000,
            int(
                get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_CAPTURE_MS", "6500")
                or "6500"
            ),
        ),
        "streaming_dual_lane_enabled": _parse_bool(
            get_value("TWINR_STREAMING_DUAL_LANE_ENABLED"), True
        ),
        "streaming_first_word_enabled": _parse_bool(
            get_value("TWINR_STREAMING_FIRST_WORD_ENABLED"), True
        ),
        "streaming_first_word_model": get_value("TWINR_STREAMING_FIRST_WORD_MODEL", "")
        or "",
        "streaming_first_word_reasoning_effort": get_value(
            "TWINR_STREAMING_FIRST_WORD_REASONING_EFFORT", ""
        )
        or "",
        "streaming_first_word_context_turns": max(
            0, int(get_value("TWINR_STREAMING_FIRST_WORD_CONTEXT_TURNS", "1") or "1")
        ),
        "streaming_first_word_max_output_tokens": max(
            16,
            int(
                get_value("TWINR_STREAMING_FIRST_WORD_MAX_OUTPUT_TOKENS", "32") or "32"
            ),
        ),
        "streaming_first_word_prefetch_enabled": _parse_bool(
            get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_ENABLED"), True
        ),
        "streaming_first_word_prefetch_min_chars": max(
            1,
            int(get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_MIN_CHARS", "4") or "4"),
        ),
        "streaming_first_word_prefetch_min_words": max(
            1,
            int(get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_MIN_WORDS", "2") or "2"),
        ),
        "streaming_first_word_prefetch_wait_ms": max(
            0,
            int(get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_WAIT_MS", "40") or "40"),
        ),
        "streaming_bridge_reply_timeout_ms": max(
            0, int(get_value("TWINR_STREAMING_BRIDGE_REPLY_TIMEOUT_MS", "250") or "250")
        ),
        "streaming_first_word_final_lane_wait_ms": max(
            0,
            int(
                get_value("TWINR_STREAMING_FIRST_WORD_FINAL_LANE_WAIT_MS", "900")
                or "900"
            ),
        ),
        "streaming_final_lane_watchdog_timeout_ms": max(
            25,
            int(
                get_value("TWINR_STREAMING_FINAL_LANE_WATCHDOG_TIMEOUT_MS", "4000")
                or "4000"
            ),
        ),
        "streaming_final_lane_hard_timeout_ms": max(
            50,
            int(
                get_value("TWINR_STREAMING_FINAL_LANE_HARD_TIMEOUT_MS", "15000")
                or "15000"
            ),
        ),
        "streaming_search_final_lane_watchdog_timeout_ms": max(
            25,
            int(
                get_value(
                    "TWINR_STREAMING_SEARCH_FINAL_LANE_WATCHDOG_TIMEOUT_MS", "6000"
                )
                or "6000"
            ),
        ),
        "streaming_search_final_lane_hard_timeout_ms": max(
            50,
            int(
                get_value("TWINR_STREAMING_SEARCH_FINAL_LANE_HARD_TIMEOUT_MS", "30000")
                or "30000"
            ),
        ),
        "streaming_supervisor_model": get_value("TWINR_STREAMING_SUPERVISOR_MODEL", "")
        or "",
        "streaming_supervisor_reasoning_effort": get_value(
            "TWINR_STREAMING_SUPERVISOR_REASONING_EFFORT", "low"
        )
        or "low",
        "streaming_supervisor_context_turns": int(
            get_value("TWINR_STREAMING_SUPERVISOR_CONTEXT_TURNS", "4") or "4"
        ),
        "streaming_supervisor_max_output_tokens": int(
            get_value("TWINR_STREAMING_SUPERVISOR_MAX_OUTPUT_TOKENS", "80") or "80"
        ),
        "streaming_supervisor_prefetch_enabled": _parse_bool(
            get_value("TWINR_STREAMING_SUPERVISOR_PREFETCH_ENABLED"), True
        ),
        "streaming_supervisor_prefetch_min_chars": int(
            get_value("TWINR_STREAMING_SUPERVISOR_PREFETCH_MIN_CHARS", "8") or "8"
        ),
        "streaming_supervisor_prefetch_wait_ms": int(
            get_value("TWINR_STREAMING_SUPERVISOR_PREFETCH_WAIT_MS", "80") or "80"
        ),
        "streaming_specialist_model": get_value("TWINR_STREAMING_SPECIALIST_MODEL", "")
        or "",
        "streaming_specialist_reasoning_effort": get_value(
            "TWINR_STREAMING_SPECIALIST_REASONING_EFFORT", "low"
        )
        or "low",
        "local_semantic_router_mode": _parse_local_semantic_router_mode(
            get_value("TWINR_LOCAL_SEMANTIC_ROUTER_MODE", "off"), "off"
        ),
        "local_semantic_router_model_dir": _parse_optional_text(
            get_value("TWINR_LOCAL_SEMANTIC_ROUTER_MODEL_DIR")
        ),
        "local_semantic_router_user_intent_model_dir": _parse_optional_text(
            get_value("TWINR_LOCAL_SEMANTIC_ROUTER_USER_INTENT_MODEL_DIR")
        ),
        "local_semantic_router_trace": _parse_bool(
            get_value("TWINR_LOCAL_SEMANTIC_ROUTER_TRACE"), True
        ),
        "conversation_follow_up_enabled": _parse_bool(
            get_value("TWINR_CONVERSATION_FOLLOW_UP_ENABLED"), False
        ),
        "conversation_follow_up_after_proactive_enabled": _parse_bool(
            get_value("TWINR_CONVERSATION_FOLLOW_UP_AFTER_PROACTIVE_ENABLED"), False
        ),
        "conversation_closure_guard_enabled": _parse_bool(
            get_value("TWINR_CONVERSATION_CLOSURE_GUARD_ENABLED"), True
        ),
        "conversation_closure_model": get_value("TWINR_CONVERSATION_CLOSURE_MODEL", "")
        or "",
        "conversation_closure_reasoning_effort": get_value(
            "TWINR_CONVERSATION_CLOSURE_REASONING_EFFORT", ""
        )
        or "",
        "conversation_closure_context_turns": int(
            get_value("TWINR_CONVERSATION_CLOSURE_CONTEXT_TURNS", "4") or "4"
        ),
        "conversation_closure_instructions_file": get_value(
            "TWINR_CONVERSATION_CLOSURE_INSTRUCTIONS_FILE", "CONVERSATION_CLOSURE.md"
        )
        or "CONVERSATION_CLOSURE.md",
        "conversation_closure_max_output_tokens": max(
            16,
            int(
                get_value("TWINR_CONVERSATION_CLOSURE_MAX_OUTPUT_TOKENS", "32") or "32"
            ),
        ),
        "conversation_closure_provider_timeout_seconds": _parse_float(
            get_value("TWINR_CONVERSATION_CLOSURE_PROVIDER_TIMEOUT_SECONDS"),
            2.0,
            minimum=0.25,
        ),
        "conversation_closure_max_transcript_chars": int(
            get_value("TWINR_CONVERSATION_CLOSURE_MAX_TRANSCRIPT_CHARS", "512") or "512"
        ),
        "conversation_closure_max_response_chars": int(
            get_value("TWINR_CONVERSATION_CLOSURE_MAX_RESPONSE_CHARS", "512") or "512"
        ),
        "conversation_closure_max_reason_chars": int(
            get_value("TWINR_CONVERSATION_CLOSURE_MAX_REASON_CHARS", "256") or "256"
        ),
        "conversation_closure_min_confidence": _parse_clamped_float(
            get_value("TWINR_CONVERSATION_CLOSURE_MIN_CONFIDENCE"),
            0.65,
            minimum=0.0,
            maximum=1.0,
        ),
        "conversation_follow_up_timeout_s": _parse_float(
            get_value("TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S"), 4.0
        ),
        "audio_beep_frequency_hz": int(
            get_value("TWINR_AUDIO_BEEP_FREQUENCY_HZ", "1046") or "1046"
        ),
        "audio_beep_duration_ms": int(
            get_value("TWINR_AUDIO_BEEP_DURATION_MS", "180") or "180"
        ),
        "audio_beep_volume": _parse_float(get_value("TWINR_AUDIO_BEEP_VOLUME"), 0.8),
        "audio_beep_settle_ms": int(
            get_value("TWINR_AUDIO_BEEP_SETTLE_MS", "120") or "120"
        ),
        "processing_feedback_delay_ms": int(
            get_value("TWINR_PROCESSING_FEEDBACK_DELAY_MS", "0") or "0"
        ),
        "search_feedback_tones_enabled": _parse_bool(
            get_value("TWINR_SEARCH_FEEDBACK_TONES_ENABLED"), True
        ),
        "search_feedback_delay_ms": int(
            get_value("TWINR_SEARCH_FEEDBACK_DELAY_MS", "1200") or "1200"
        ),
        "search_feedback_pause_ms": int(
            get_value("TWINR_SEARCH_FEEDBACK_PAUSE_MS", "1100") or "1100"
        ),
        "search_feedback_volume": _parse_float(
            get_value("TWINR_SEARCH_FEEDBACK_VOLUME"), 0.09
        ),
        "audio_dynamic_pause_enabled": _parse_bool(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_ENABLED"), True
        ),
        "audio_dynamic_pause_short_utterance_max_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_UTTERANCE_MAX_MS", "1000")
            or "1000"
        ),
        "audio_dynamic_pause_long_utterance_min_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_LONG_UTTERANCE_MIN_MS", "5000")
            or "5000"
        ),
        "audio_dynamic_pause_short_pause_bonus_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_PAUSE_BONUS_MS", "120") or "120"
        ),
        "audio_dynamic_pause_short_pause_grace_bonus_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_PAUSE_GRACE_BONUS_MS", "0")
            or "0"
        ),
        "audio_dynamic_pause_medium_pause_penalty_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_MEDIUM_PAUSE_PENALTY_MS", "120")
            or "120"
        ),
        "audio_dynamic_pause_medium_pause_grace_penalty_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_MEDIUM_PAUSE_GRACE_PENALTY_MS", "250")
            or "250"
        ),
        "audio_dynamic_pause_long_pause_penalty_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_LONG_PAUSE_PENALTY_MS", "320") or "320"
        ),
        "audio_dynamic_pause_long_pause_grace_penalty_ms": int(
            get_value("TWINR_AUDIO_DYNAMIC_PAUSE_LONG_PAUSE_GRACE_PENALTY_MS", "220")
            or "220"
        ),
        "audio_pause_resume_chunks": int(
            get_value("TWINR_AUDIO_PAUSE_RESUME_CHUNKS", "2") or "2"
        ),
        "audio_speech_start_chunks": int(
            get_value("TWINR_AUDIO_SPEECH_START_CHUNKS", "1") or "1"
        ),
        "audio_follow_up_speech_start_chunks": int(
            get_value("TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS", "1") or "1"
        ),
        "audio_follow_up_ignore_ms": int(
            get_value("TWINR_AUDIO_FOLLOW_UP_IGNORE_MS", "0") or "0"
        ),
        "openai_enable_web_search": _parse_bool(
            get_value("TWINR_OPENAI_ENABLE_WEB_SEARCH"), False
        ),
        "openai_search_model": get_value("OPENAI_SEARCH_MODEL", "") or "",
        "openai_web_search_context_size": get_value(
            "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE", "medium"
        )
        or "medium",
        "openai_search_max_output_tokens": int(
            get_value("TWINR_OPENAI_SEARCH_MAX_OUTPUT_TOKENS", "1024") or "1024"
        ),
        "openai_search_retry_max_output_tokens": int(
            get_value("TWINR_OPENAI_SEARCH_RETRY_MAX_OUTPUT_TOKENS", "1536") or "1536"
        ),
        "openai_web_search_country": get_value("TWINR_OPENAI_WEB_SEARCH_COUNTRY"),
        "openai_web_search_region": get_value("TWINR_OPENAI_WEB_SEARCH_REGION"),
        "openai_web_search_city": get_value("TWINR_OPENAI_WEB_SEARCH_CITY"),
        "openai_web_search_timezone": get_value("TWINR_OPENAI_WEB_SEARCH_TIMEZONE"),
        "conversation_web_search": (
            get_value("TWINR_CONVERSATION_WEB_SEARCH", "auto") or "auto"
        )
        .strip()
        .lower(),
        "audio_input_device": get_value("TWINR_AUDIO_INPUT_DEVICE", "default")
        or "default",
        "audio_output_device": get_value("TWINR_AUDIO_OUTPUT_DEVICE", "default")
        or "default",
        "audio_sample_rate": int(
            get_value("TWINR_AUDIO_SAMPLE_RATE", "16000") or "16000"
        ),
        "audio_channels": int(get_value("TWINR_AUDIO_CHANNELS", "1") or "1"),
        "audio_chunk_ms": int(get_value("TWINR_AUDIO_CHUNK_MS", "100") or "100"),
        "audio_preroll_ms": int(get_value("TWINR_AUDIO_PREROLL_MS", "300") or "300"),
        "audio_speech_threshold": int(
            get_value("TWINR_AUDIO_SPEECH_THRESHOLD", "700") or "700"
        ),
        "audio_start_timeout_s": _parse_float(
            get_value("TWINR_AUDIO_START_TIMEOUT_S"), 8.0
        ),
        "audio_max_record_seconds": _parse_float(
            get_value("TWINR_AUDIO_MAX_RECORD_SECONDS"), 20.0
        ),
        "streaming_tts_clause_min_chars": int(
            get_value("TWINR_STREAMING_TTS_CLAUSE_MIN_CHARS", "28") or "28"
        ),
        "streaming_tts_soft_segment_chars": int(
            get_value("TWINR_STREAMING_TTS_SOFT_SEGMENT_CHARS", "72") or "72"
        ),
        "streaming_tts_hard_segment_chars": int(
            get_value("TWINR_STREAMING_TTS_HARD_SEGMENT_CHARS", "120") or "120"
        ),
        "openai_tts_stream_chunk_size": int(
            get_value("OPENAI_TTS_STREAM_CHUNK_SIZE", "2048") or "2048"
        ),
        "tts_worker_join_timeout_s": _parse_float(
            get_value("TWINR_TTS_WORKER_JOIN_TIMEOUT_S"), 60.0
        ),
    }
