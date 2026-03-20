"""Define the canonical environment-backed runtime configuration for Twinr.

``TwinrConfig`` is the single source of truth for provider selection, runtime
timing, hardware wiring, memory settings, and operator-facing service ports.
Load it through ``TwinrConfig.from_env()`` instead of duplicating `.env`
parsing in adjacent modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import os

DEFAULT_BUTTON_PROBE_LINES = (
    4,
    5,
    6,
    12,
    13,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
)
DEFAULT_WAKEWORD_PHRASES = (
    "hey twinr",
    "he twinr",
    "hey twinna",
    "hey twina",
    "hey twinner",
    "hallo twinr",
    "hallo twinna",
    "hallo twina",
    "hallo twinner",
    "twinr hallo",
    "twinr hey",
    "twinna hallo",
    "twinna hey",
    "twina hallo",
    "twina hey",
    "twinner hallo",
    "twinner hey",
    "twinr",
    "twinna",
    "twina",
    "twinner",
)
# Custom openWakeWord models can legitimately need very low operating thresholds.
# Keep the parser permissive and let deployment tuning decide the actual value.
MIN_SAFE_OPENWAKEWORD_THRESHOLD = 0.0
SUPPORTED_DISPLAY_DRIVERS = (
    "hdmi_wayland",
    "hdmi_fbdev",
    "waveshare_4in2_v2",
)
GPIO_DISPLAY_DRIVERS = frozenset({"waveshare_4in2_v2"})
SUPPORTED_DISPLAY_LAYOUTS = (
    "default",
    "debug_log",
    "debug_face",
)


def _read_dotenv(path: Path) -> dict[str, str]:
    """Read simple ``KEY=VALUE`` pairs from a dotenv-style file."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a Twinr boolean env value with a fallback default."""

    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _parse_optional_bool(value: str | None) -> bool | None:
    """Parse an optional boolean env value or return ``None``."""

    if value is None or not value.strip():
        return None
    return _parse_bool(value, False)


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer env value or return ``None``."""

    if value is None or not value.strip():
        return None
    return int(value)


def _parse_float(
    value: str | float | int | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a float env value and clamp it to optional bounds."""

    if value is None:
        parsed = default
    elif isinstance(value, str):
        if not value.strip():
            parsed = default
        else:
            parsed = float(value)
    else:
        parsed = float(value)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _parse_clamped_float(value: str | None, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    """Parse a float env value through the shared clamp-aware helper."""

    return _parse_float(value, default, minimum=minimum, maximum=maximum)


def _parse_csv_ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse a comma-separated integer list or return the default tuple."""

    if value is None or not value.strip():
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv_strings(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated string list or return the default tuple."""

    if value is None or not value.strip():
        return default
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    return parsed or default


def _parse_csv_mapping(
    value: str | None,
    default: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Parse one comma-separated ``key=value`` mapping into a normalized tuple."""

    if value is None or not value.strip():
        return default
    parsed: list[tuple[str, str]] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Expected KEY=VALUE entry, got {part!r}.")
        key, mapped_value = (piece.strip() for piece in part.split("=", 1))
        if not key or not mapped_value:
            raise ValueError(f"Expected non-empty KEY=VALUE entry, got {part!r}.")
        parsed.append((key, mapped_value))
    return tuple(parsed) or default


def _default_bundled_openwakeword_models(project_root: Path) -> tuple[str, ...]:
    """Return the bundled Twinr wakeword model when the leading repo ships it."""

    candidate = (
        project_root / "src" / "twinr" / "proactive" / "wakeword" / "models" / "twinr_v1.onnx"
    ).resolve(strict=False)
    if candidate.exists():
        return (str(candidate),)
    return ()


def _default_bundled_openwakeword_custom_verifier_models(
    project_root: Path,
    models: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return bundled verifier assets that live next to configured local models."""

    bundled: list[tuple[str, str]] = []
    for raw_model in models:
        model = str(raw_model).strip()
        if not model or not model.lower().endswith((".onnx", ".tflite")):
            continue
        candidate = Path(model).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        resolved_model = candidate.resolve(strict=False)
        verifier_candidate = resolved_model.with_suffix(".verifier.pkl")
        if verifier_candidate.exists():
            bundled.append((resolved_model.stem, str(verifier_candidate)))
    return tuple(bundled)


def _default_openwakeword_inference_framework(models: tuple[str, ...]) -> str:
    """Infer the local runtime from the configured wakeword model paths."""

    normalized_models = tuple(str(item).strip().lower() for item in models if str(item).strip())
    if normalized_models and all(item.endswith(".onnx") for item in normalized_models):
        return "onnx"
    if normalized_models and all(item.endswith(".tflite") for item in normalized_models):
        return "tflite"
    return "tflite"


@dataclass(frozen=True, slots=True)
class TwinrConfig:
    """Store the immutable runtime settings snapshot for the base agent.

    The dataclass groups provider selection, streaming and wakeword tuning,
    hardware wiring, memory durability paths, proactive sensing thresholds,
    and operator service endpoints into one canonical object.
    """

    openai_api_key: str | None = None
    openai_project_id: str | None = None
    openai_send_project_header: bool | None = None
    stt_provider: str = "openai"
    llm_provider: str = "openai"
    tts_provider: str = "openai"
    project_root: str = "."
    personality_dir: str = "personality"
    user_display_name: str | None = None
    default_model: str = "gpt-5.2"
    openai_reasoning_effort: str = "medium"
    openai_prompt_cache_enabled: bool = True
    openai_prompt_cache_retention: str | None = None
    openai_stt_model: str = "whisper-1"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "marin"
    openai_tts_speed: float = 1.0
    openai_tts_format: str = "wav"
    openai_tts_instructions: str | None = None
    deepgram_api_key: str | None = None
    deepgram_base_url: str = "https://api.deepgram.com/v1"
    deepgram_stt_model: str = "nova-3"
    deepgram_stt_language: str | None = "de"
    deepgram_stt_smart_format: bool = True
    deepgram_streaming_interim_results: bool = True
    deepgram_streaming_endpointing_ms: int = 400
    deepgram_streaming_utterance_end_ms: int = 1000
    deepgram_streaming_stop_on_utterance_end: bool = True
    deepgram_streaming_finalize_timeout_s: float = 4.0
    deepgram_timeout_s: float = 30.0
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_s: float = 45.0
    openai_realtime_model: str = "gpt-4o-realtime-preview"
    openai_realtime_voice: str = "sage"
    openai_realtime_speed: float = 1.0
    openai_realtime_instructions: str | None = None
    openai_realtime_transcription_model: str = "whisper-1"
    openai_realtime_language: str | None = "de"
    openai_realtime_input_sample_rate: int = 24000
    turn_controller_enabled: bool = True
    turn_controller_context_turns: int = 4
    turn_controller_instructions_file: str = "TURN_CONTROLLER.md"
    turn_controller_fast_endpoint_enabled: bool = True
    turn_controller_fast_endpoint_min_chars: int = 10
    turn_controller_fast_endpoint_min_confidence: float = 0.9
    turn_controller_backchannel_max_chars: int = 24
    turn_controller_interrupt_enabled: bool = True
    turn_controller_interrupt_window_ms: int = 420
    turn_controller_interrupt_poll_ms: int = 120
    turn_controller_interrupt_min_active_ratio: float = 0.18
    turn_controller_interrupt_min_transcript_chars: int = 4
    turn_controller_interrupt_consecutive_windows: int = 2
    streaming_early_transcript_enabled: bool = True
    streaming_early_transcript_min_chars: int = 10
    streaming_early_transcript_wait_ms: int = 250
    streaming_transcript_verifier_enabled: bool = True
    streaming_transcript_verifier_model: str = "gpt-4o-mini-transcribe"
    streaming_transcript_verifier_max_words: int = 6
    streaming_transcript_verifier_max_chars: int = 32
    streaming_transcript_verifier_min_confidence: float = 0.92
    streaming_transcript_verifier_max_capture_ms: int = 6500
    streaming_dual_lane_enabled: bool = True
    streaming_first_word_enabled: bool = True
    streaming_first_word_model: str = "gpt-4o-mini"
    streaming_first_word_reasoning_effort: str = ""
    streaming_first_word_context_turns: int = 1
    streaming_first_word_max_output_tokens: int = 32
    streaming_first_word_prefetch_enabled: bool = True
    streaming_first_word_prefetch_min_chars: int = 4
    streaming_first_word_prefetch_min_words: int = 2
    streaming_first_word_prefetch_wait_ms: int = 40
    streaming_bridge_reply_timeout_ms: int = 250
    streaming_first_word_final_lane_wait_ms: int = 900
    streaming_final_lane_watchdog_timeout_ms: int = 4000
    streaming_final_lane_hard_timeout_ms: int = 15000
    streaming_supervisor_model: str = "gpt-4o-mini"
    streaming_supervisor_reasoning_effort: str = "low"
    streaming_supervisor_context_turns: int = 4
    streaming_supervisor_max_output_tokens: int = 80
    streaming_supervisor_prefetch_enabled: bool = True
    streaming_supervisor_prefetch_min_chars: int = 8
    streaming_supervisor_prefetch_wait_ms: int = 80
    streaming_specialist_model: str | None = "gpt-4o-mini"
    streaming_specialist_reasoning_effort: str | None = "low"
    conversation_follow_up_enabled: bool = False
    conversation_follow_up_after_proactive_enabled: bool = False
    conversation_closure_guard_enabled: bool = True
    conversation_closure_context_turns: int = 4
    conversation_closure_instructions_file: str = "CONVERSATION_CLOSURE.md"
    conversation_closure_provider_timeout_seconds: float = 2.0
    conversation_closure_max_transcript_chars: int = 512
    conversation_closure_max_response_chars: int = 512
    conversation_closure_max_reason_chars: int = 256
    conversation_closure_min_confidence: float = 0.65
    conversation_follow_up_timeout_s: float = 4.0
    audio_beep_frequency_hz: int = 1046
    audio_beep_duration_ms: int = 180
    audio_beep_volume: float = 0.8
    audio_beep_settle_ms: int = 120
    processing_feedback_delay_ms: int = 0
    search_feedback_tones_enabled: bool = True
    search_feedback_delay_ms: int = 1200
    search_feedback_pause_ms: int = 900
    search_feedback_volume: float = 0.14
    audio_dynamic_pause_enabled: bool = True
    audio_dynamic_pause_short_utterance_max_ms: int = 1000
    audio_dynamic_pause_long_utterance_min_ms: int = 5000
    audio_dynamic_pause_short_pause_bonus_ms: int = 120
    audio_dynamic_pause_short_pause_grace_bonus_ms: int = 0
    audio_dynamic_pause_medium_pause_penalty_ms: int = 120
    audio_dynamic_pause_medium_pause_grace_penalty_ms: int = 250
    audio_dynamic_pause_long_pause_penalty_ms: int = 320
    audio_dynamic_pause_long_pause_grace_penalty_ms: int = 220
    audio_pause_resume_chunks: int = 2
    audio_speech_start_chunks: int = 1
    audio_follow_up_speech_start_chunks: int = 1
    audio_follow_up_ignore_ms: int = 0
    openai_enable_web_search: bool = False
    openai_search_model: str = "gpt-4o-mini-search-preview"
    openai_web_search_context_size: str = "medium"
    openai_search_max_output_tokens: int = 160
    openai_search_retry_max_output_tokens: int = 240
    openai_web_search_country: str | None = None
    openai_web_search_region: str | None = None
    openai_web_search_city: str | None = None
    openai_web_search_timezone: str | None = None
    conversation_web_search: str = "auto"
    audio_input_device: str = "default"
    audio_output_device: str = "default"
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_chunk_ms: int = 100
    audio_preroll_ms: int = 300
    audio_speech_threshold: int = 700
    audio_start_timeout_s: float = 8.0
    audio_max_record_seconds: float = 20.0
    streaming_tts_clause_min_chars: int = 28
    streaming_tts_soft_segment_chars: int = 72
    streaming_tts_hard_segment_chars: int = 120
    openai_tts_stream_chunk_size: int = 2048
    tts_worker_join_timeout_s: float = 60.0
    orchestrator_host: str = "0.0.0.0"
    orchestrator_port: int = 8797
    orchestrator_ws_url: str = "ws://127.0.0.1:8797/ws/orchestrator"
    orchestrator_shared_secret: str | None = None
    whatsapp_node_binary: str = "node"
    whatsapp_allow_from: str | None = None
    whatsapp_auth_dir: str = "state/channels/whatsapp/auth"
    whatsapp_worker_root: str = "src/twinr/channels/whatsapp/worker"
    whatsapp_groups_enabled: bool = False
    whatsapp_self_chat_mode: bool = False
    whatsapp_reconnect_base_delay_s: float = 2.0
    whatsapp_reconnect_max_delay_s: float = 30.0
    whatsapp_send_timeout_s: float = 20.0
    whatsapp_sent_cache_ttl_s: float = 180.0
    whatsapp_sent_cache_max_entries: int = 256
    camera_device: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_framerate: int = 30
    camera_input_format: str | None = None
    camera_ffmpeg_path: str = "ffmpeg"
    vision_reference_image_path: str | None = None
    portrait_match_enabled: bool = True
    portrait_match_detector_model_path: str = "state/opencv/models/face_detection_yunet_2023mar.onnx"
    portrait_match_recognizer_model_path: str = "state/opencv/models/face_recognition_sface_2021dec.onnx"
    portrait_match_likely_threshold: float = 0.45
    portrait_match_uncertain_threshold: float = 0.34
    portrait_match_max_age_s: float = 45.0
    portrait_match_capture_lock_timeout_s: float = 5.0
    portrait_match_store_path: str = "state/portrait_identities.json"
    portrait_match_reference_image_dir: str = "state/portrait_identities"
    portrait_match_primary_user_id: str = "main_user"
    portrait_match_max_reference_images_per_user: int = 6
    portrait_match_identity_margin: float = 0.05
    portrait_match_temporal_window_s: float = 300.0
    portrait_match_temporal_min_observations: int = 2
    portrait_match_temporal_max_observations: int = 12
    openai_vision_detail: str = "auto"
    proactive_enabled: bool = False
    proactive_vision_provider: str = "local_first"
    proactive_local_camera_detection_network_path: str = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    proactive_local_camera_pose_network_path: str = "/usr/share/imx500-models/imx500_network_posenet.rpk"
    proactive_local_camera_pose_backend: str = "mediapipe"
    proactive_local_camera_mediapipe_pose_model_path: str = "state/mediapipe/models/pose_landmarker_full.task"
    proactive_local_camera_mediapipe_hand_landmarker_model_path: str = "state/mediapipe/models/hand_landmarker.task"
    proactive_local_camera_mediapipe_gesture_model_path: str = "state/mediapipe/models/gesture_recognizer.task"
    proactive_local_camera_mediapipe_custom_gesture_model_path: str | None = None
    proactive_local_camera_mediapipe_num_hands: int = 2
    proactive_local_camera_sequence_window_s: float = 1.6
    proactive_local_camera_sequence_min_frames: int = 4
    proactive_local_camera_source_device: str = "imx500"
    proactive_local_camera_frame_rate: int = 10
    proactive_local_camera_lock_timeout_s: float = 5.0
    proactive_local_camera_startup_warmup_s: float = 0.8
    proactive_local_camera_metadata_wait_s: float = 3.0
    proactive_local_camera_person_confidence_threshold: float = 0.40
    proactive_local_camera_object_confidence_threshold: float = 0.55
    proactive_local_camera_person_near_area_threshold: float = 0.20
    proactive_local_camera_person_near_height_threshold: float = 0.55
    proactive_local_camera_object_near_area_threshold: float = 0.08
    proactive_local_camera_attention_score_threshold: float = 0.62
    proactive_local_camera_engaged_score_threshold: float = 0.45
    proactive_local_camera_pose_confidence_threshold: float = 0.30
    proactive_local_camera_pose_refresh_s: float = 12.0
    proactive_poll_interval_s: float = 4.0
    proactive_capture_interval_s: float = 6.0
    proactive_motion_window_s: float = 20.0
    proactive_low_motion_after_s: float = 12.0
    proactive_audio_enabled: bool = False
    proactive_audio_input_device: str | None = None
    proactive_audio_sample_ms: int = 1000
    proactive_audio_distress_enabled: bool = False
    proactive_vision_review_enabled: bool = False
    proactive_vision_review_buffer_frames: int = 8
    proactive_vision_review_max_frames: int = 4
    proactive_vision_review_max_age_s: float = 12.0
    proactive_vision_review_min_spacing_s: float = 1.2
    wakeword_enabled: bool = False
    wakeword_backend: str = "openwakeword"
    wakeword_primary_backend: str = "openwakeword"
    wakeword_fallback_backend: str = "stt"
    wakeword_verifier_mode: str = "ambiguity_only"
    wakeword_verifier_margin: float = 0.08
    wakeword_phrases: tuple[str, ...] = DEFAULT_WAKEWORD_PHRASES
    wakeword_stt_phrases: tuple[str, ...] = DEFAULT_WAKEWORD_PHRASES
    wakeword_sample_ms: int = 1800
    wakeword_presence_grace_s: float = 15.0 * 60.0
    wakeword_motion_grace_s: float = 5.0 * 60.0
    wakeword_speech_grace_s: float = 90.0
    wakeword_attempt_cooldown_s: float = 4.0
    wakeword_block_proactive_after_attempt_s: float = 20.0
    wakeword_min_active_ratio: float = 0.08
    wakeword_min_active_chunks: int = 2
    wakeword_openwakeword_models: tuple[str, ...] = ()
    wakeword_openwakeword_custom_verifier_models: tuple[tuple[str, str], ...] = ()
    wakeword_openwakeword_custom_verifier_threshold: float = 0.1
    wakeword_openwakeword_threshold: float = 0.5
    wakeword_openwakeword_vad_threshold: float = 0.0
    wakeword_openwakeword_patience_frames: int = 1
    wakeword_openwakeword_activation_samples: int = 3
    wakeword_openwakeword_deactivation_threshold: float = 0.2
    wakeword_openwakeword_enable_speex: bool = False
    wakeword_openwakeword_transcribe_on_detect: bool = False
    wakeword_openwakeword_inference_framework: str = "tflite"
    wakeword_calibration_profile_path: str = "state/wakeword_calibration.json"
    wakeword_calibration_recommended_path: str = "state/wakeword_calibration.recommended.json"
    proactive_person_returned_absence_s: float = 20.0 * 60.0
    proactive_person_returned_recent_motion_s: float = 30.0
    proactive_attention_window_s: float = 6.0
    proactive_slumped_quiet_s: float = 20.0
    proactive_possible_fall_stillness_s: float = 10.0
    proactive_possible_fall_visibility_loss_hold_s: float = 15.0
    proactive_possible_fall_visibility_loss_arming_s: float = 6.0
    proactive_possible_fall_slumped_visibility_loss_arming_s: float = 4.0
    proactive_possible_fall_once_per_presence_session: bool = True
    proactive_floor_stillness_s: float = 20.0
    proactive_showing_intent_hold_s: float = 1.5
    proactive_positive_contact_hold_s: float = 1.5
    proactive_distress_hold_s: float = 3.0
    proactive_fall_transition_window_s: float = 8.0
    proactive_person_returned_score_threshold: float = 0.9
    proactive_attention_window_score_threshold: float = 0.86
    proactive_slumped_quiet_score_threshold: float = 0.9
    proactive_possible_fall_score_threshold: float = 0.82
    proactive_floor_stillness_score_threshold: float = 0.9
    proactive_showing_intent_score_threshold: float = 0.84
    proactive_positive_contact_score_threshold: float = 0.84
    proactive_distress_possible_score_threshold: float = 0.85
    proactive_governor_enabled: bool = True
    proactive_governor_active_reservation_ttl_s: float = 45.0
    proactive_governor_global_prompt_cooldown_s: float = 120.0
    proactive_governor_window_s: float = 20.0 * 60.0
    proactive_governor_window_prompt_limit: int = 4
    proactive_governor_presence_session_prompt_limit: int = 2
    proactive_governor_source_repeat_cooldown_s: float = 10.0 * 60.0
    proactive_governor_history_limit: int = 128
    proactive_visual_first_audio_global_cooldown_s: float = 5.0 * 60.0
    proactive_visual_first_audio_source_repeat_cooldown_s: float = 15.0 * 60.0
    proactive_visual_first_cue_hold_s: float = 45.0
    proactive_quiet_hours_visual_only_enabled: bool = True
    proactive_quiet_hours_start_local: str = "21:00"
    proactive_quiet_hours_end_local: str = "07:00"
    web_host: str = "0.0.0.0"
    web_port: int = 1337
    runtime_state_path: str = "/tmp/twinr-runtime-state.json"
    memory_markdown_path: str = "state/MEMORY.md"
    reminder_store_path: str = "state/reminders.json"
    automation_store_path: str = "state/automations.json"
    voice_profile_store_path: str = "state/voice_profile.json"
    adaptive_timing_enabled: bool = True
    adaptive_timing_store_path: str = "state/adaptive_timing.json"
    adaptive_timing_pause_grace_ms: int = 450
    long_term_memory_enabled: bool = False
    long_term_memory_backend: str = "chonkydb"
    long_term_memory_mode: str = "local_first"
    long_term_memory_remote_required: bool = False
    long_term_memory_remote_namespace: str | None = None
    long_term_memory_path: str = "state/chonkydb"
    long_term_memory_background_store_turns: bool = True
    long_term_memory_write_queue_size: int = 32
    long_term_memory_recall_limit: int = 3
    long_term_memory_query_rewrite_enabled: bool = True
    long_term_memory_remote_read_timeout_s: float = 8.0
    long_term_memory_remote_write_timeout_s: float = 15.0
    long_term_memory_remote_keepalive_interval_s: float = 5.0
    long_term_memory_remote_runtime_check_mode: str = "direct"
    long_term_memory_remote_watchdog_startup_wait_s: float = 30.0
    long_term_memory_remote_watchdog_interval_s: float = 1.0
    long_term_memory_remote_watchdog_history_limit: int = 3600
    long_term_memory_remote_max_content_chars: int = 2_000_000
    long_term_memory_remote_shard_max_content_chars: int = 1_000_000
    long_term_memory_remote_retry_attempts: int = 3
    long_term_memory_remote_retry_backoff_s: float = 1.0
    long_term_memory_remote_flush_timeout_s: float = 60.0
    long_term_memory_remote_read_cache_ttl_s: float = 0.0
    long_term_memory_turn_extractor_model: str | None = None
    long_term_memory_turn_extractor_max_output_tokens: int = 2200
    long_term_memory_midterm_enabled: bool = True
    long_term_memory_midterm_limit: int = 4
    long_term_memory_reflection_window_size: int = 18
    long_term_memory_reflection_compiler_enabled: bool = True
    long_term_memory_reflection_compiler_model: str | None = None
    long_term_memory_reflection_compiler_max_output_tokens: int = 900
    long_term_memory_subtext_compiler_enabled: bool = True
    long_term_memory_subtext_compiler_model: str | None = None
    long_term_memory_subtext_compiler_max_output_tokens: int = 520
    long_term_memory_proactive_enabled: bool = False
    long_term_memory_proactive_poll_interval_s: float = 30.0
    long_term_memory_proactive_min_confidence: float = 0.72
    long_term_memory_proactive_repeat_cooldown_s: float = 6.0 * 60.0 * 60.0
    long_term_memory_proactive_skip_cooldown_s: float = 30.0 * 60.0
    long_term_memory_proactive_reservation_ttl_s: float = 90.0
    long_term_memory_proactive_allow_sensitive: bool = False
    long_term_memory_proactive_history_limit: int = 128
    long_term_memory_sensor_memory_enabled: bool = False
    long_term_memory_sensor_baseline_days: int = 21
    long_term_memory_sensor_min_days_observed: int = 6
    long_term_memory_sensor_min_routine_ratio: float = 0.55
    long_term_memory_sensor_deviation_min_delta: float = 0.45
    long_term_memory_retention_enabled: bool = True
    long_term_memory_retention_mode: str = "conservative"
    long_term_memory_retention_run_interval_s: float = 300.0
    long_term_memory_archive_enabled: bool = True
    long_term_memory_migration_enabled: bool = True
    long_term_memory_migration_batch_size: int = 64
    long_term_memory_remote_bulk_request_max_bytes: int = 512 * 1024
    chonkydb_base_url: str | None = None
    chonkydb_api_key: str | None = None
    chonkydb_api_key_header: str = "x-api-key"
    chonkydb_allow_bearer_auth: bool = False
    chonkydb_timeout_s: float = 20.0
    chonkydb_max_response_bytes: int = 64 * 1024 * 1024
    restore_runtime_state_on_startup: bool = False
    reminder_poll_interval_s: float = 1.0
    reminder_retry_delay_s: float = 90.0
    reminder_max_entries: int = 48
    automation_poll_interval_s: float = 5.0
    automation_max_entries: int = 96
    voice_profile_min_sample_ms: int = 1200
    voice_profile_likely_threshold: float = 0.72
    voice_profile_uncertain_threshold: float = 0.55
    voice_profile_max_samples: int = 6
    speech_pause_ms: int = 1200
    memory_max_turns: int = 20
    memory_keep_recent: int = 10
    gpio_chip: str = "gpiochip0"
    green_button_gpio: int | None = None
    yellow_button_gpio: int | None = None
    pir_motion_gpio: int | None = None
    pir_active_high: bool = True
    pir_bias: str = "pull-down"
    pir_debounce_ms: int = 120
    button_active_low: bool = True
    button_bias: str = "pull-up"
    button_debounce_ms: int = 80
    button_probe_lines: tuple[int, ...] = DEFAULT_BUTTON_PROBE_LINES
    display_driver: str = "hdmi_fbdev"
    display_companion_enabled: bool | None = None
    display_fb_path: str = "/dev/fb0"
    display_wayland_display: str = "wayland-0"
    display_wayland_runtime_dir: str | None = None
    display_face_cue_path: str = "artifacts/stores/ops/display_face_cue.json"
    display_face_cue_ttl_s: float = 4.0
    display_attention_refresh_interval_s: float = 1.25
    display_presentation_path: str = "artifacts/stores/ops/display_presentation.json"
    display_presentation_ttl_s: float = 20.0
    display_vendor_dir: str = "state/display/vendor"
    display_spi_bus: int = 0
    display_spi_device: int = 0
    display_cs_gpio: int = 8
    display_dc_gpio: int = 25
    display_reset_gpio: int = 17
    display_busy_gpio: int = 24
    display_width: int = 400
    display_height: int = 300
    display_rotation_degrees: int = 270
    display_full_refresh_interval: int = 0
    display_busy_timeout_s: float = 20.0
    display_runtime_trace_enabled: bool = False
    display_poll_interval_s: float = 0.5
    display_layout: str = "default"
    display_news_ticker_enabled: bool = False
    display_news_ticker_feed_urls: tuple[str, ...] = ()
    display_news_ticker_store_path: str = "artifacts/stores/ops/display_news_ticker.json"
    display_news_ticker_refresh_interval_s: float = 900.0
    display_news_ticker_rotation_interval_s: float = 12.0
    display_news_ticker_max_items: int = 12
    display_news_ticker_timeout_s: float = 4.0
    printer_queue: str = "Thermal_GP58"
    printer_device_uri: str | None = None
    printer_header_text: str = "TWINR.com"
    printer_feed_lines: int = 3
    printer_line_width: int = 30
    print_button_cooldown_s: float = 2.0
    print_max_lines: int = 8
    print_max_chars: int = 320
    print_context_turns: int = 6

    def __post_init__(self) -> None:
        """Normalize derived long-term-memory mode fields after construction."""

        normalized_mode = str(self.long_term_memory_mode or "local_first").strip().lower() or "local_first"
        normalized_display_driver = str(self.display_driver or "hdmi_fbdev").strip().lower() or "hdmi_fbdev"
        normalized_display_layout = str(self.display_layout or "default").strip().lower() or "default"
        if normalized_display_layout == "debug_face":
            normalized_display_layout = "debug_log"
        if normalized_display_driver not in SUPPORTED_DISPLAY_DRIVERS:
            raise ValueError(
                "display_driver must be one of: "
                + ", ".join(SUPPORTED_DISPLAY_DRIVERS)
            )
        if normalized_display_layout not in SUPPORTED_DISPLAY_LAYOUTS:
            raise ValueError(
                "display_layout must be one of: "
                + ", ".join(SUPPORTED_DISPLAY_LAYOUTS)
            )
        normalized_display_busy_timeout_s = float(self.display_busy_timeout_s)
        if not math.isfinite(normalized_display_busy_timeout_s):
            raise ValueError("display_busy_timeout_s must be finite")
        normalized_display_busy_timeout_s = max(0.1, normalized_display_busy_timeout_s)
        normalized_display_face_cue_ttl_s = float(self.display_face_cue_ttl_s)
        if not math.isfinite(normalized_display_face_cue_ttl_s):
            raise ValueError("display_face_cue_ttl_s must be finite")
        normalized_display_face_cue_ttl_s = max(0.1, normalized_display_face_cue_ttl_s)
        normalized_display_attention_refresh_interval_s = float(self.display_attention_refresh_interval_s)
        if not math.isfinite(normalized_display_attention_refresh_interval_s):
            raise ValueError("display_attention_refresh_interval_s must be finite")
        normalized_display_attention_refresh_interval_s = max(0.0, normalized_display_attention_refresh_interval_s)
        normalized_display_presentation_ttl_s = float(self.display_presentation_ttl_s)
        if not math.isfinite(normalized_display_presentation_ttl_s):
            raise ValueError("display_presentation_ttl_s must be finite")
        normalized_display_presentation_ttl_s = max(0.1, normalized_display_presentation_ttl_s)
        normalized_display_news_ticker_refresh_interval_s = float(self.display_news_ticker_refresh_interval_s)
        if not math.isfinite(normalized_display_news_ticker_refresh_interval_s):
            raise ValueError("display_news_ticker_refresh_interval_s must be finite")
        normalized_display_news_ticker_refresh_interval_s = max(30.0, normalized_display_news_ticker_refresh_interval_s)
        normalized_display_news_ticker_rotation_interval_s = float(self.display_news_ticker_rotation_interval_s)
        if not math.isfinite(normalized_display_news_ticker_rotation_interval_s):
            raise ValueError("display_news_ticker_rotation_interval_s must be finite")
        normalized_display_news_ticker_rotation_interval_s = max(4.0, normalized_display_news_ticker_rotation_interval_s)
        normalized_display_news_ticker_timeout_s = float(self.display_news_ticker_timeout_s)
        if not math.isfinite(normalized_display_news_ticker_timeout_s):
            raise ValueError("display_news_ticker_timeout_s must be finite")
        normalized_display_news_ticker_timeout_s = max(0.5, normalized_display_news_ticker_timeout_s)
        normalized_display_news_ticker_max_items = max(1, int(self.display_news_ticker_max_items))
        normalized_display_face_cue_path = (
            str(self.display_face_cue_path or "artifacts/stores/ops/display_face_cue.json").strip()
            or "artifacts/stores/ops/display_face_cue.json"
        )
        normalized_display_presentation_path = (
            str(self.display_presentation_path or "artifacts/stores/ops/display_presentation.json").strip()
            or "artifacts/stores/ops/display_presentation.json"
        )
        normalized_proactive_quiet_hours_start_local = (
            str(self.proactive_quiet_hours_start_local or "21:00").strip() or "21:00"
        )
        normalized_proactive_quiet_hours_end_local = (
            str(self.proactive_quiet_hours_end_local or "07:00").strip() or "07:00"
        )
        normalized_display_news_ticker_store_path = (
            str(self.display_news_ticker_store_path or "artifacts/stores/ops/display_news_ticker.json").strip()
            or "artifacts/stores/ops/display_news_ticker.json"
        )
        object.__setattr__(self, "long_term_memory_mode", normalized_mode)
        object.__setattr__(
            self,
            "long_term_memory_remote_required",
            normalized_mode == "remote_primary",
        )
        object.__setattr__(self, "display_driver", normalized_display_driver)
        object.__setattr__(self, "display_busy_timeout_s", normalized_display_busy_timeout_s)
        object.__setattr__(self, "display_face_cue_path", normalized_display_face_cue_path)
        object.__setattr__(self, "display_face_cue_ttl_s", normalized_display_face_cue_ttl_s)
        object.__setattr__(
            self,
            "display_attention_refresh_interval_s",
            normalized_display_attention_refresh_interval_s,
        )
        object.__setattr__(self, "display_presentation_path", normalized_display_presentation_path)
        object.__setattr__(self, "display_presentation_ttl_s", normalized_display_presentation_ttl_s)
        object.__setattr__(self, "display_news_ticker_store_path", normalized_display_news_ticker_store_path)
        object.__setattr__(
            self,
            "display_news_ticker_refresh_interval_s",
            normalized_display_news_ticker_refresh_interval_s,
        )
        object.__setattr__(
            self,
            "display_news_ticker_rotation_interval_s",
            normalized_display_news_ticker_rotation_interval_s,
        )
        object.__setattr__(self, "display_news_ticker_max_items", normalized_display_news_ticker_max_items)
        object.__setattr__(self, "display_news_ticker_timeout_s", normalized_display_news_ticker_timeout_s)
        object.__setattr__(self, "display_layout", normalized_display_layout)
        object.__setattr__(self, "proactive_quiet_hours_start_local", normalized_proactive_quiet_hours_start_local)
        object.__setattr__(self, "proactive_quiet_hours_end_local", normalized_proactive_quiet_hours_end_local)

    @property
    def button_gpios(self) -> dict[str, int]:
        """Return the configured button GPIO mapping keyed by button name."""

        mapping: dict[str, int] = {}
        if self.green_button_gpio is not None:
            mapping["green"] = self.green_button_gpio
        if self.yellow_button_gpio is not None:
            mapping["yellow"] = self.yellow_button_gpio
        return mapping

    @property
    def display_gpios(self) -> dict[str, int]:
        """Return the configured display GPIO assignments with operator labels."""

        if not self.display_uses_gpio:
            return {}
        return {
            "Display CS": self.display_cs_gpio,
            "Display DC": self.display_dc_gpio,
            "Display RESET": self.display_reset_gpio,
            "Display BUSY": self.display_busy_gpio,
        }

    @property
    def display_uses_gpio(self) -> bool:
        """Return whether the active display driver requires display GPIO pins."""

        return self.display_driver in GPIO_DISPLAY_DRIVERS

    @property
    def supported_display_drivers(self) -> tuple[str, ...]:
        """Return the supported display driver identifiers."""

        return SUPPORTED_DISPLAY_DRIVERS

    def display_gpio_conflicts(self) -> tuple[str, ...]:
        """Report detected GPIO collisions between display and other inputs."""

        if not self.display_uses_gpio:
            return ()
        assignments: list[tuple[str, int]] = list(self.display_gpios.items())
        if self.green_button_gpio is not None:
            assignments.append(("green button", self.green_button_gpio))
        if self.yellow_button_gpio is not None:
            assignments.append(("yellow button", self.yellow_button_gpio))
        if self.pir_motion_gpio is not None:
            assignments.append(("PIR sensor", self.pir_motion_gpio))

        labels_by_line: dict[int, list[str]] = {}
        for label, line in assignments:
            labels_by_line.setdefault(line, []).append(label)

        conflicts: list[str] = []
        for line, labels in sorted(labels_by_line.items()):
            if len(labels) < 2:
                continue
            display_labels = [label for label in labels if label.startswith("Display ")]
            other_labels = [label for label in labels if not label.startswith("Display ")]
            if other_labels:
                for display_label in display_labels:
                    for other_label in other_labels:
                        conflicts.append(
                            f"{display_label} GPIO {line} collides with {other_label} GPIO {line}."
                        )
                continue
            for index, left in enumerate(display_labels):
                for right in display_labels[index + 1 :]:
                    conflicts.append(f"{left} GPIO {line} collides with {right} GPIO {line}.")
        return tuple(conflicts)

    @property
    def pir_enabled(self) -> bool:
        """Return whether PIR motion sensing is configured."""

        return self.pir_motion_gpio is not None

    @property
    def local_timezone_name(self) -> str:
        """Return the configured local timezone name with a stable fallback."""

        return (self.openai_web_search_timezone or "Europe/Berlin").strip() or "Europe/Berlin"

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "TwinrConfig":
        """Build a config snapshot from process env and a dotenv file.

        Environment variables override values from ``env_path``. Missing values
        fall back to the defaults defined on ``TwinrConfig``.

        Args:
            env_path: Dotenv path to read before applying process-environment
                overrides. Defaults to ``.env``.

        Returns:
            A fully populated immutable ``TwinrConfig`` instance.
        """

        path = Path(env_path)
        file_values = _read_dotenv(path)
        project_root = path.parent.resolve()
        default_remote_runtime_check_mode = (
            "watchdog_artifact" if project_root == Path("/twinr") else "direct"
        )

        def get_value(name: str, default: str | None = None) -> str | None:
            if name in os.environ:
                return os.environ[name]
            return file_values.get(name, default)

        wakeword_primary_backend = (
            get_value(
                "TWINR_WAKEWORD_PRIMARY_BACKEND",
                get_value("TWINR_WAKEWORD_BACKEND", "openwakeword"),
            )
            or "openwakeword"
        ).strip().lower()
        wakeword_verifier_mode = (
            get_value("TWINR_WAKEWORD_VERIFIER_MODE")
            or (
                "always"
                if _parse_bool(get_value("TWINR_WAKEWORD_OPENWAKEWORD_TRANSCRIBE_ON_DETECT"), False)
                else "ambiguity_only"
            )
        ).strip().lower()
        bundled_openwakeword_models = _default_bundled_openwakeword_models(project_root)
        wakeword_phrases = _parse_csv_strings(
            get_value("TWINR_WAKEWORD_PHRASES"),
            DEFAULT_WAKEWORD_PHRASES,
        )
        wakeword_stt_phrases = _parse_csv_strings(
            get_value("TWINR_WAKEWORD_STT_PHRASES"),
            wakeword_phrases,
        )
        wakeword_openwakeword_models = _parse_csv_strings(
            get_value("TWINR_WAKEWORD_OPENWAKEWORD_MODELS"),
            bundled_openwakeword_models,
        )
        wakeword_openwakeword_custom_verifier_models = _parse_csv_mapping(
            get_value("TWINR_WAKEWORD_OPENWAKEWORD_CUSTOM_VERIFIER_MODELS"),
            _default_bundled_openwakeword_custom_verifier_models(
                project_root,
                wakeword_openwakeword_models,
            ),
        )
        wakeword_openwakeword_inference_framework = (
            get_value("TWINR_WAKEWORD_OPENWAKEWORD_INFERENCE_FRAMEWORK")
            or _default_openwakeword_inference_framework(wakeword_openwakeword_models)
        ).strip().lower()

        return cls(
            openai_api_key=get_value("OPENAI_API_KEY"),
            openai_project_id=get_value("OPENAI_PROJ_ID"),
            openai_send_project_header=_parse_optional_bool(get_value("OPENAI_SEND_PROJECT_HEADER")),
            stt_provider=(get_value("TWINR_STT_PROVIDER", "openai") or "openai").strip().lower(),
            llm_provider=(get_value("TWINR_LLM_PROVIDER", "openai") or "openai").strip().lower(),
            tts_provider=(get_value("TWINR_TTS_PROVIDER", "openai") or "openai").strip().lower(),
            project_root=str(project_root),
            personality_dir=get_value("TWINR_PERSONALITY_DIR", "personality") or "personality",
            user_display_name=get_value("TWINR_USER_DISPLAY_NAME"),
            default_model=get_value("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
            openai_reasoning_effort=get_value("OPENAI_REASONING_EFFORT", "medium") or "medium",
            openai_prompt_cache_enabled=_parse_bool(get_value("OPENAI_PROMPT_CACHE_ENABLED"), True),
            openai_prompt_cache_retention=get_value("OPENAI_PROMPT_CACHE_RETENTION"),
            openai_stt_model=get_value("OPENAI_STT_MODEL", "whisper-1") or "whisper-1",
            openai_tts_model=get_value("OPENAI_TTS_MODEL", "gpt-4o-mini-tts") or "gpt-4o-mini-tts",
            openai_tts_voice=get_value("OPENAI_TTS_VOICE", "marin") or "marin",
            openai_tts_speed=_parse_float(get_value("OPENAI_TTS_SPEED"), 1.0),
            openai_tts_format=get_value("OPENAI_TTS_FORMAT", "wav") or "wav",
            openai_tts_instructions=get_value("OPENAI_TTS_INSTRUCTIONS"),
            deepgram_api_key=get_value("DEEPGRAM_API_KEY"),
            deepgram_base_url=get_value("DEEPGRAM_BASE_URL", "https://api.deepgram.com/v1")
            or "https://api.deepgram.com/v1",
            deepgram_stt_model=get_value("DEEPGRAM_STT_MODEL", "nova-3") or "nova-3",
            deepgram_stt_language=get_value("DEEPGRAM_STT_LANGUAGE", "de") or "de",
            deepgram_stt_smart_format=_parse_bool(get_value("DEEPGRAM_STT_SMART_FORMAT"), True),
            deepgram_streaming_interim_results=_parse_bool(
                get_value("DEEPGRAM_STREAMING_INTERIM_RESULTS"),
                True,
            ),
            deepgram_streaming_endpointing_ms=int(
                get_value("DEEPGRAM_STREAMING_ENDPOINTING_MS", "400") or "400"
            ),
            deepgram_streaming_utterance_end_ms=int(
                get_value("DEEPGRAM_STREAMING_UTTERANCE_END_MS", "1000") or "1000"
            ),
            deepgram_streaming_stop_on_utterance_end=_parse_bool(
                get_value("DEEPGRAM_STREAMING_STOP_ON_UTTERANCE_END"),
                True,
            ),
            deepgram_streaming_finalize_timeout_s=_parse_float(
                get_value("DEEPGRAM_STREAMING_FINALIZE_TIMEOUT_S"),
                4.0,
            ),
            deepgram_timeout_s=_parse_float(get_value("DEEPGRAM_TIMEOUT_S"), 30.0),
            groq_api_key=get_value("GROQ_API_KEY"),
            groq_base_url=get_value("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            or "https://api.groq.com/openai/v1",
            groq_model=get_value("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile",
            groq_timeout_s=_parse_float(get_value("GROQ_TIMEOUT_S"), 45.0),
            openai_realtime_model=get_value("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
            or "gpt-4o-realtime-preview",
            openai_realtime_voice=get_value("OPENAI_REALTIME_VOICE", "sage") or "sage",
            openai_realtime_speed=_parse_float(get_value("OPENAI_REALTIME_SPEED"), 1.0),
            openai_realtime_instructions=get_value("OPENAI_REALTIME_INSTRUCTIONS"),
            openai_realtime_transcription_model=(
                get_value("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "whisper-1") or "whisper-1"
            ),
            openai_realtime_language=get_value("OPENAI_REALTIME_LANGUAGE", "de") or "de",
            openai_realtime_input_sample_rate=int(
                get_value("OPENAI_REALTIME_INPUT_SAMPLE_RATE", "24000") or "24000"
            ),
            turn_controller_enabled=_parse_bool(
                get_value("TWINR_TURN_CONTROLLER_ENABLED"),
                True,
            ),
            turn_controller_context_turns=int(
                get_value("TWINR_TURN_CONTROLLER_CONTEXT_TURNS", "4") or "4"
            ),
            turn_controller_instructions_file=(
                get_value("TWINR_TURN_CONTROLLER_INSTRUCTIONS_FILE", "TURN_CONTROLLER.md")
                or "TURN_CONTROLLER.md"
            ),
            turn_controller_fast_endpoint_enabled=_parse_bool(
                get_value("TWINR_TURN_CONTROLLER_FAST_ENDPOINT_ENABLED"),
                True,
            ),
            turn_controller_fast_endpoint_min_chars=int(
                get_value("TWINR_TURN_CONTROLLER_FAST_ENDPOINT_MIN_CHARS", "10") or "10"
            ),
            turn_controller_fast_endpoint_min_confidence=_parse_clamped_float(
                get_value("TWINR_TURN_CONTROLLER_FAST_ENDPOINT_MIN_CONFIDENCE"),
                0.9,
                minimum=0.0,
                maximum=1.0,
            ),
            turn_controller_backchannel_max_chars=int(
                get_value("TWINR_TURN_CONTROLLER_BACKCHANNEL_MAX_CHARS", "24") or "24"
            ),
            turn_controller_interrupt_enabled=_parse_bool(
                get_value("TWINR_TURN_CONTROLLER_INTERRUPT_ENABLED"),
                True,
            ),
            turn_controller_interrupt_window_ms=int(
                get_value("TWINR_TURN_CONTROLLER_INTERRUPT_WINDOW_MS", "420") or "420"
            ),
            turn_controller_interrupt_poll_ms=int(
                get_value("TWINR_TURN_CONTROLLER_INTERRUPT_POLL_MS", "120") or "120"
            ),
            turn_controller_interrupt_min_active_ratio=_parse_clamped_float(
                get_value("TWINR_TURN_CONTROLLER_INTERRUPT_MIN_ACTIVE_RATIO"),
                0.18,
                minimum=0.0,
                maximum=1.0,
            ),
            turn_controller_interrupt_min_transcript_chars=int(
                get_value("TWINR_TURN_CONTROLLER_INTERRUPT_MIN_TRANSCRIPT_CHARS", "4") or "4"
            ),
            turn_controller_interrupt_consecutive_windows=int(
                get_value("TWINR_TURN_CONTROLLER_INTERRUPT_CONSECUTIVE_WINDOWS", "2") or "2"
            ),
            streaming_early_transcript_enabled=_parse_bool(
                get_value("TWINR_STREAMING_EARLY_TRANSCRIPT_ENABLED"),
                True,
            ),
            streaming_early_transcript_min_chars=int(
                get_value("TWINR_STREAMING_EARLY_TRANSCRIPT_MIN_CHARS", "10") or "10"
            ),
            streaming_early_transcript_wait_ms=int(
                get_value("TWINR_STREAMING_EARLY_TRANSCRIPT_WAIT_MS", "250") or "250"
            ),
            streaming_transcript_verifier_enabled=_parse_bool(
                get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_ENABLED"),
                True,
            ),
            streaming_transcript_verifier_model=(
                get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MODEL", "gpt-4o-mini-transcribe")
                or "gpt-4o-mini-transcribe"
            ),
            streaming_transcript_verifier_max_words=max(
                1,
                int(get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_WORDS", "6") or "6"),
            ),
            streaming_transcript_verifier_max_chars=max(
                8,
                int(get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_CHARS", "32") or "32"),
            ),
            streaming_transcript_verifier_min_confidence=_parse_clamped_float(
                get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MIN_CONFIDENCE"),
                0.92,
                minimum=0.0,
                maximum=1.0,
            ),
            streaming_transcript_verifier_max_capture_ms=max(
                1000,
                int(get_value("TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_CAPTURE_MS", "6500") or "6500"),
            ),
            streaming_dual_lane_enabled=_parse_bool(
                get_value("TWINR_STREAMING_DUAL_LANE_ENABLED"),
                True,
            ),
            streaming_first_word_enabled=_parse_bool(
                get_value("TWINR_STREAMING_FIRST_WORD_ENABLED"),
                True,
            ),
            streaming_first_word_model=(
                get_value("TWINR_STREAMING_FIRST_WORD_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
            ),
            streaming_first_word_reasoning_effort=(
                get_value("TWINR_STREAMING_FIRST_WORD_REASONING_EFFORT", "") or ""
            ),
            streaming_first_word_context_turns=max(
                0,
                int(get_value("TWINR_STREAMING_FIRST_WORD_CONTEXT_TURNS", "1") or "1"),
            ),
            streaming_first_word_max_output_tokens=max(
                16,
                int(get_value("TWINR_STREAMING_FIRST_WORD_MAX_OUTPUT_TOKENS", "32") or "32"),
            ),
            streaming_first_word_prefetch_enabled=_parse_bool(
                get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_ENABLED"),
                True,
            ),
            streaming_first_word_prefetch_min_chars=max(
                1,
                int(get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_MIN_CHARS", "4") or "4"),
            ),
            streaming_first_word_prefetch_min_words=max(
                1,
                int(get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_MIN_WORDS", "2") or "2"),
            ),
            streaming_first_word_prefetch_wait_ms=max(
                0,
                int(get_value("TWINR_STREAMING_FIRST_WORD_PREFETCH_WAIT_MS", "40") or "40"),
            ),
            streaming_bridge_reply_timeout_ms=max(
                0,
                int(get_value("TWINR_STREAMING_BRIDGE_REPLY_TIMEOUT_MS", "250") or "250"),
            ),
            streaming_first_word_final_lane_wait_ms=max(
                0,
                int(get_value("TWINR_STREAMING_FIRST_WORD_FINAL_LANE_WAIT_MS", "900") or "900"),
            ),
            streaming_final_lane_watchdog_timeout_ms=max(
                25,
                int(get_value("TWINR_STREAMING_FINAL_LANE_WATCHDOG_TIMEOUT_MS", "4000") or "4000"),
            ),
            streaming_final_lane_hard_timeout_ms=max(
                50,
                int(get_value("TWINR_STREAMING_FINAL_LANE_HARD_TIMEOUT_MS", "15000") or "15000"),
            ),
            streaming_supervisor_model=(
                get_value("TWINR_STREAMING_SUPERVISOR_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
            ),
            streaming_supervisor_reasoning_effort=(
                get_value("TWINR_STREAMING_SUPERVISOR_REASONING_EFFORT", "low") or "low"
            ),
            streaming_supervisor_context_turns=int(
                get_value("TWINR_STREAMING_SUPERVISOR_CONTEXT_TURNS", "4") or "4"
            ),
            streaming_supervisor_max_output_tokens=int(
                get_value("TWINR_STREAMING_SUPERVISOR_MAX_OUTPUT_TOKENS", "80") or "80"
            ),
            streaming_supervisor_prefetch_enabled=_parse_bool(
                get_value("TWINR_STREAMING_SUPERVISOR_PREFETCH_ENABLED"),
                True,
            ),
            streaming_supervisor_prefetch_min_chars=int(
                get_value("TWINR_STREAMING_SUPERVISOR_PREFETCH_MIN_CHARS", "8") or "8"
            ),
            streaming_supervisor_prefetch_wait_ms=int(
                get_value("TWINR_STREAMING_SUPERVISOR_PREFETCH_WAIT_MS", "80") or "80"
            ),
            streaming_specialist_model=(
                get_value("TWINR_STREAMING_SPECIALIST_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
            ),
            streaming_specialist_reasoning_effort=(
                get_value("TWINR_STREAMING_SPECIALIST_REASONING_EFFORT", "low") or "low"
            ),
            conversation_follow_up_enabled=_parse_bool(
                get_value("TWINR_CONVERSATION_FOLLOW_UP_ENABLED"),
                False,
            ),
            conversation_follow_up_after_proactive_enabled=_parse_bool(
                get_value("TWINR_CONVERSATION_FOLLOW_UP_AFTER_PROACTIVE_ENABLED"),
                False,
            ),
            conversation_closure_guard_enabled=_parse_bool(
                get_value("TWINR_CONVERSATION_CLOSURE_GUARD_ENABLED"),
                True,
            ),
            conversation_closure_context_turns=int(
                get_value("TWINR_CONVERSATION_CLOSURE_CONTEXT_TURNS", "4") or "4"
            ),
            conversation_closure_instructions_file=(
                get_value("TWINR_CONVERSATION_CLOSURE_INSTRUCTIONS_FILE", "CONVERSATION_CLOSURE.md")
                or "CONVERSATION_CLOSURE.md"
            ),
            conversation_closure_provider_timeout_seconds=_parse_float(
                get_value("TWINR_CONVERSATION_CLOSURE_PROVIDER_TIMEOUT_SECONDS"),
                2.0,
                minimum=0.25,
            ),
            conversation_closure_max_transcript_chars=int(
                get_value("TWINR_CONVERSATION_CLOSURE_MAX_TRANSCRIPT_CHARS", "512") or "512"
            ),
            conversation_closure_max_response_chars=int(
                get_value("TWINR_CONVERSATION_CLOSURE_MAX_RESPONSE_CHARS", "512") or "512"
            ),
            conversation_closure_max_reason_chars=int(
                get_value("TWINR_CONVERSATION_CLOSURE_MAX_REASON_CHARS", "256") or "256"
            ),
            conversation_closure_min_confidence=_parse_clamped_float(
                get_value("TWINR_CONVERSATION_CLOSURE_MIN_CONFIDENCE"),
                0.65,
                minimum=0.0,
                maximum=1.0,
            ),
            conversation_follow_up_timeout_s=_parse_float(
                get_value("TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S"),
                4.0,
            ),
            audio_beep_frequency_hz=int(get_value("TWINR_AUDIO_BEEP_FREQUENCY_HZ", "1046") or "1046"),
            audio_beep_duration_ms=int(get_value("TWINR_AUDIO_BEEP_DURATION_MS", "180") or "180"),
            audio_beep_volume=_parse_float(get_value("TWINR_AUDIO_BEEP_VOLUME"), 0.8),
            audio_beep_settle_ms=int(get_value("TWINR_AUDIO_BEEP_SETTLE_MS", "120") or "120"),
            processing_feedback_delay_ms=int(
                get_value("TWINR_PROCESSING_FEEDBACK_DELAY_MS", "0") or "0"
            ),
            search_feedback_tones_enabled=_parse_bool(get_value("TWINR_SEARCH_FEEDBACK_TONES_ENABLED"), True),
            search_feedback_delay_ms=int(get_value("TWINR_SEARCH_FEEDBACK_DELAY_MS", "1200") or "1200"),
            search_feedback_pause_ms=int(get_value("TWINR_SEARCH_FEEDBACK_PAUSE_MS", "900") or "900"),
            search_feedback_volume=_parse_float(get_value("TWINR_SEARCH_FEEDBACK_VOLUME"), 0.14),
            audio_dynamic_pause_enabled=_parse_bool(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_ENABLED"),
                True,
            ),
            audio_dynamic_pause_short_utterance_max_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_UTTERANCE_MAX_MS", "1000") or "1000"
            ),
            audio_dynamic_pause_long_utterance_min_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_LONG_UTTERANCE_MIN_MS", "5000") or "5000"
            ),
            audio_dynamic_pause_short_pause_bonus_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_PAUSE_BONUS_MS", "120") or "120"
            ),
            audio_dynamic_pause_short_pause_grace_bonus_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_PAUSE_GRACE_BONUS_MS", "0") or "0"
            ),
            audio_dynamic_pause_medium_pause_penalty_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_MEDIUM_PAUSE_PENALTY_MS", "120") or "120"
            ),
            audio_dynamic_pause_medium_pause_grace_penalty_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_MEDIUM_PAUSE_GRACE_PENALTY_MS", "250") or "250"
            ),
            audio_dynamic_pause_long_pause_penalty_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_LONG_PAUSE_PENALTY_MS", "320") or "320"
            ),
            audio_dynamic_pause_long_pause_grace_penalty_ms=int(
                get_value("TWINR_AUDIO_DYNAMIC_PAUSE_LONG_PAUSE_GRACE_PENALTY_MS", "220") or "220"
            ),
            audio_pause_resume_chunks=int(
                get_value("TWINR_AUDIO_PAUSE_RESUME_CHUNKS", "2") or "2"
            ),
            audio_speech_start_chunks=int(get_value("TWINR_AUDIO_SPEECH_START_CHUNKS", "1") or "1"),
            audio_follow_up_speech_start_chunks=int(
                get_value("TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS", "1") or "1"
            ),
            audio_follow_up_ignore_ms=int(get_value("TWINR_AUDIO_FOLLOW_UP_IGNORE_MS", "0") or "0"),
            openai_enable_web_search=_parse_bool(get_value("TWINR_OPENAI_ENABLE_WEB_SEARCH"), False),
            openai_search_model=get_value("OPENAI_SEARCH_MODEL", "gpt-4o-mini-search-preview")
            or "gpt-4o-mini-search-preview",
            openai_web_search_context_size=get_value("TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE", "medium") or "medium",
            openai_search_max_output_tokens=int(
                get_value("TWINR_OPENAI_SEARCH_MAX_OUTPUT_TOKENS", "160") or "160"
            ),
            openai_search_retry_max_output_tokens=int(
                get_value("TWINR_OPENAI_SEARCH_RETRY_MAX_OUTPUT_TOKENS", "240") or "240"
            ),
            openai_web_search_country=get_value("TWINR_OPENAI_WEB_SEARCH_COUNTRY"),
            openai_web_search_region=get_value("TWINR_OPENAI_WEB_SEARCH_REGION"),
            openai_web_search_city=get_value("TWINR_OPENAI_WEB_SEARCH_CITY"),
            openai_web_search_timezone=get_value("TWINR_OPENAI_WEB_SEARCH_TIMEZONE"),
            conversation_web_search=(get_value("TWINR_CONVERSATION_WEB_SEARCH", "auto") or "auto").strip().lower(),
            audio_input_device=get_value("TWINR_AUDIO_INPUT_DEVICE", "default") or "default",
            audio_output_device=get_value("TWINR_AUDIO_OUTPUT_DEVICE", "default") or "default",
            audio_sample_rate=int(get_value("TWINR_AUDIO_SAMPLE_RATE", "16000") or "16000"),
            audio_channels=int(get_value("TWINR_AUDIO_CHANNELS", "1") or "1"),
            audio_chunk_ms=int(get_value("TWINR_AUDIO_CHUNK_MS", "100") or "100"),
            audio_preroll_ms=int(get_value("TWINR_AUDIO_PREROLL_MS", "300") or "300"),
            audio_speech_threshold=int(get_value("TWINR_AUDIO_SPEECH_THRESHOLD", "700") or "700"),
            audio_start_timeout_s=_parse_float(get_value("TWINR_AUDIO_START_TIMEOUT_S"), 8.0),
            audio_max_record_seconds=_parse_float(get_value("TWINR_AUDIO_MAX_RECORD_SECONDS"), 20.0),
            streaming_tts_clause_min_chars=int(
                get_value("TWINR_STREAMING_TTS_CLAUSE_MIN_CHARS", "28") or "28"
            ),
            streaming_tts_soft_segment_chars=int(
                get_value("TWINR_STREAMING_TTS_SOFT_SEGMENT_CHARS", "72") or "72"
            ),
            streaming_tts_hard_segment_chars=int(
                get_value("TWINR_STREAMING_TTS_HARD_SEGMENT_CHARS", "120") or "120"
            ),
            openai_tts_stream_chunk_size=int(
                get_value("OPENAI_TTS_STREAM_CHUNK_SIZE", "2048") or "2048"
            ),
            tts_worker_join_timeout_s=_parse_float(
                get_value("TWINR_TTS_WORKER_JOIN_TIMEOUT_S"),
                60.0,
            ),
            orchestrator_host=get_value("TWINR_ORCHESTRATOR_HOST", "0.0.0.0") or "0.0.0.0",
            orchestrator_port=int(get_value("TWINR_ORCHESTRATOR_PORT", "8797") or "8797"),
            orchestrator_ws_url=(
                get_value("TWINR_ORCHESTRATOR_WS_URL", "ws://127.0.0.1:8797/ws/orchestrator")
                or "ws://127.0.0.1:8797/ws/orchestrator"
            ),
            orchestrator_shared_secret=get_value("TWINR_ORCHESTRATOR_SHARED_SECRET") or None,
            whatsapp_node_binary=get_value("TWINR_WHATSAPP_NODE_BINARY", "node") or "node",
            whatsapp_allow_from=get_value("TWINR_WHATSAPP_ALLOW_FROM") or None,
            whatsapp_auth_dir=get_value(
                "TWINR_WHATSAPP_AUTH_DIR",
                str(project_root / "state" / "channels" / "whatsapp" / "auth"),
            )
            or str(project_root / "state" / "channels" / "whatsapp" / "auth"),
            whatsapp_worker_root=get_value(
                "TWINR_WHATSAPP_WORKER_ROOT",
                str(project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"),
            )
            or str(project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"),
            whatsapp_groups_enabled=_parse_bool(get_value("TWINR_WHATSAPP_GROUPS_ENABLED"), False),
            whatsapp_self_chat_mode=_parse_bool(get_value("TWINR_WHATSAPP_SELF_CHAT_MODE"), False),
            whatsapp_reconnect_base_delay_s=_parse_float(
                get_value("TWINR_WHATSAPP_RECONNECT_BASE_DELAY_S"),
                2.0,
                minimum=0.1,
            ),
            whatsapp_reconnect_max_delay_s=_parse_float(
                get_value("TWINR_WHATSAPP_RECONNECT_MAX_DELAY_S"),
                30.0,
                minimum=0.1,
            ),
            whatsapp_send_timeout_s=_parse_float(
                get_value("TWINR_WHATSAPP_SEND_TIMEOUT_S"),
                20.0,
                minimum=1.0,
            ),
            whatsapp_sent_cache_ttl_s=_parse_float(
                get_value("TWINR_WHATSAPP_SENT_CACHE_TTL_S"),
                180.0,
                minimum=1.0,
            ),
            whatsapp_sent_cache_max_entries=max(
                16,
                int(get_value("TWINR_WHATSAPP_SENT_CACHE_MAX_ENTRIES", "256") or "256"),
            ),
            camera_device=get_value("TWINR_CAMERA_DEVICE", "/dev/video0") or "/dev/video0",
            camera_width=int(get_value("TWINR_CAMERA_WIDTH", "640") or "640"),
            camera_height=int(get_value("TWINR_CAMERA_HEIGHT", "480") or "480"),
            camera_framerate=int(get_value("TWINR_CAMERA_FRAMERATE", "30") or "30"),
            camera_input_format=get_value("TWINR_CAMERA_INPUT_FORMAT"),
            camera_ffmpeg_path=get_value("TWINR_CAMERA_FFMPEG_PATH", "ffmpeg") or "ffmpeg",
            vision_reference_image_path=get_value("TWINR_VISION_REFERENCE_IMAGE"),
            portrait_match_enabled=_parse_bool(get_value("TWINR_PORTRAIT_MATCH_ENABLED"), True),
            portrait_match_detector_model_path=(
                get_value(
                    "TWINR_PORTRAIT_MATCH_DETECTOR_MODEL_PATH",
                    "state/opencv/models/face_detection_yunet_2023mar.onnx",
                )
                or "state/opencv/models/face_detection_yunet_2023mar.onnx"
            ),
            portrait_match_recognizer_model_path=(
                get_value(
                    "TWINR_PORTRAIT_MATCH_RECOGNIZER_MODEL_PATH",
                    "state/opencv/models/face_recognition_sface_2021dec.onnx",
                )
                or "state/opencv/models/face_recognition_sface_2021dec.onnx"
            ),
            portrait_match_likely_threshold=_parse_clamped_float(
                get_value("TWINR_PORTRAIT_MATCH_LIKELY_THRESHOLD"),
                0.45,
                minimum=0.0,
                maximum=1.0,
            ),
            portrait_match_uncertain_threshold=_parse_clamped_float(
                get_value("TWINR_PORTRAIT_MATCH_UNCERTAIN_THRESHOLD"),
                0.34,
                minimum=0.0,
                maximum=1.0,
            ),
            portrait_match_max_age_s=_parse_float(
                get_value("TWINR_PORTRAIT_MATCH_MAX_AGE_S"),
                45.0,
                minimum=0.0,
            ),
            portrait_match_capture_lock_timeout_s=_parse_float(
                get_value("TWINR_PORTRAIT_MATCH_CAPTURE_LOCK_TIMEOUT_S"),
                5.0,
                minimum=0.0,
            ),
            portrait_match_store_path=(
                get_value("TWINR_PORTRAIT_MATCH_STORE_PATH", "state/portrait_identities.json")
                or "state/portrait_identities.json"
            ),
            portrait_match_reference_image_dir=(
                get_value("TWINR_PORTRAIT_MATCH_REFERENCE_IMAGE_DIR", "state/portrait_identities")
                or "state/portrait_identities"
            ),
            portrait_match_primary_user_id=(
                get_value("TWINR_PORTRAIT_MATCH_PRIMARY_USER_ID", "main_user")
                or "main_user"
            ),
            portrait_match_max_reference_images_per_user=max(
                1,
                int(get_value("TWINR_PORTRAIT_MATCH_MAX_REFERENCE_IMAGES_PER_USER", "6") or "6"),
            ),
            portrait_match_identity_margin=_parse_clamped_float(
                get_value("TWINR_PORTRAIT_MATCH_IDENTITY_MARGIN"),
                0.05,
                minimum=0.0,
                maximum=1.0,
            ),
            portrait_match_temporal_window_s=_parse_float(
                get_value("TWINR_PORTRAIT_MATCH_TEMPORAL_WINDOW_S"),
                300.0,
                minimum=0.0,
            ),
            portrait_match_temporal_min_observations=max(
                1,
                int(get_value("TWINR_PORTRAIT_MATCH_TEMPORAL_MIN_OBSERVATIONS", "2") or "2"),
            ),
            portrait_match_temporal_max_observations=max(
                1,
                int(get_value("TWINR_PORTRAIT_MATCH_TEMPORAL_MAX_OBSERVATIONS", "12") or "12"),
            ),
            openai_vision_detail=get_value("OPENAI_VISION_DETAIL", "auto") or "auto",
            proactive_enabled=_parse_bool(get_value("TWINR_PROACTIVE_ENABLED"), False),
            proactive_vision_provider=(get_value("TWINR_PROACTIVE_VISION_PROVIDER", "local_first") or "local_first").strip().lower(),
            proactive_local_camera_detection_network_path=(
                get_value(
                    "TWINR_PROACTIVE_LOCAL_CAMERA_DETECTION_NETWORK_PATH",
                    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
                )
                or "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
            ),
            proactive_local_camera_pose_network_path=(
                get_value(
                    "TWINR_PROACTIVE_LOCAL_CAMERA_POSE_NETWORK_PATH",
                    "/usr/share/imx500-models/imx500_network_posenet.rpk",
                )
                or "/usr/share/imx500-models/imx500_network_posenet.rpk"
            ),
            proactive_local_camera_pose_backend=(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_POSE_BACKEND", "mediapipe") or "mediapipe"
            ).strip().lower(),
            proactive_local_camera_mediapipe_pose_model_path=(
                get_value(
                    "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_POSE_MODEL_PATH",
                    "state/mediapipe/models/pose_landmarker_full.task",
                )
                or "state/mediapipe/models/pose_landmarker_full.task"
            ),
            proactive_local_camera_mediapipe_hand_landmarker_model_path=(
                get_value(
                    "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_HAND_LANDMARKER_MODEL_PATH",
                    "state/mediapipe/models/hand_landmarker.task",
                )
                or "state/mediapipe/models/hand_landmarker.task"
            ),
            proactive_local_camera_mediapipe_gesture_model_path=(
                get_value(
                    "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_GESTURE_MODEL_PATH",
                    "state/mediapipe/models/gesture_recognizer.task",
                )
                or "state/mediapipe/models/gesture_recognizer.task"
            ),
            proactive_local_camera_mediapipe_custom_gesture_model_path=(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_CUSTOM_GESTURE_MODEL_PATH") or None
            ),
            proactive_local_camera_mediapipe_num_hands=int(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_NUM_HANDS", "2") or "2"
            ),
            proactive_local_camera_sequence_window_s=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_SEQUENCE_WINDOW_S"),
                1.6,
            ),
            proactive_local_camera_sequence_min_frames=int(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_SEQUENCE_MIN_FRAMES", "4") or "4"
            ),
            proactive_local_camera_source_device=(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_SOURCE_DEVICE", "imx500") or "imx500"
            ),
            proactive_local_camera_frame_rate=int(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_FRAME_RATE", "10") or "10"
            ),
            proactive_local_camera_lock_timeout_s=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_LOCK_TIMEOUT_S"),
                5.0,
            ),
            proactive_local_camera_startup_warmup_s=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_STARTUP_WARMUP_S"),
                0.8,
            ),
            proactive_local_camera_metadata_wait_s=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_METADATA_WAIT_S"),
                3.0,
            ),
            proactive_local_camera_person_confidence_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PERSON_CONFIDENCE_THRESHOLD"),
                0.40,
            ),
            proactive_local_camera_object_confidence_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_OBJECT_CONFIDENCE_THRESHOLD"),
                0.55,
            ),
            proactive_local_camera_person_near_area_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PERSON_NEAR_AREA_THRESHOLD"),
                0.20,
            ),
            proactive_local_camera_person_near_height_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_PERSON_NEAR_HEIGHT_THRESHOLD"),
                0.55,
            ),
            proactive_local_camera_object_near_area_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_OBJECT_NEAR_AREA_THRESHOLD"),
                0.08,
            ),
            proactive_local_camera_attention_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_ATTENTION_SCORE_THRESHOLD"),
                0.62,
            ),
            proactive_local_camera_engaged_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_ENGAGED_SCORE_THRESHOLD"),
                0.45,
            ),
            proactive_local_camera_pose_confidence_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_POSE_CONFIDENCE_THRESHOLD"),
                0.30,
            ),
            proactive_local_camera_pose_refresh_s=_parse_float(
                get_value("TWINR_PROACTIVE_LOCAL_CAMERA_POSE_REFRESH_S"),
                12.0,
            ),
            proactive_poll_interval_s=_parse_float(get_value("TWINR_PROACTIVE_POLL_INTERVAL_S"), 4.0),
            proactive_capture_interval_s=_parse_float(get_value("TWINR_PROACTIVE_CAPTURE_INTERVAL_S"), 6.0),
            proactive_motion_window_s=_parse_float(get_value("TWINR_PROACTIVE_MOTION_WINDOW_S"), 20.0),
            proactive_low_motion_after_s=_parse_float(get_value("TWINR_PROACTIVE_LOW_MOTION_AFTER_S"), 12.0),
            proactive_audio_enabled=_parse_bool(get_value("TWINR_PROACTIVE_AUDIO_ENABLED"), False),
            proactive_audio_input_device=get_value("TWINR_PROACTIVE_AUDIO_DEVICE"),
            proactive_audio_sample_ms=int(get_value("TWINR_PROACTIVE_AUDIO_SAMPLE_MS", "1000") or "1000"),
            proactive_audio_distress_enabled=_parse_bool(
                get_value("TWINR_PROACTIVE_AUDIO_DISTRESS_ENABLED"),
                False,
            ),
            proactive_vision_review_enabled=_parse_bool(
                get_value("TWINR_PROACTIVE_VISION_REVIEW_ENABLED"),
                False,
            ),
            proactive_vision_review_buffer_frames=int(
                get_value("TWINR_PROACTIVE_VISION_REVIEW_BUFFER_FRAMES", "8") or "8"
            ),
            proactive_vision_review_max_frames=int(
                get_value("TWINR_PROACTIVE_VISION_REVIEW_MAX_FRAMES", "4") or "4"
            ),
            proactive_vision_review_max_age_s=_parse_float(
                get_value("TWINR_PROACTIVE_VISION_REVIEW_MAX_AGE_S"),
                12.0,
            ),
            proactive_vision_review_min_spacing_s=_parse_float(
                get_value("TWINR_PROACTIVE_VISION_REVIEW_MIN_SPACING_S"),
                1.2,
            ),
            wakeword_enabled=_parse_bool(get_value("TWINR_WAKEWORD_ENABLED"), False),
            wakeword_backend=wakeword_primary_backend,
            wakeword_primary_backend=wakeword_primary_backend,
            wakeword_fallback_backend=(
                get_value("TWINR_WAKEWORD_FALLBACK_BACKEND", "stt") or "stt"
            ).strip().lower(),
            wakeword_verifier_mode=wakeword_verifier_mode,
            wakeword_verifier_margin=_parse_clamped_float(
                get_value("TWINR_WAKEWORD_VERIFIER_MARGIN"),
                0.08,
                minimum=0.0,
                maximum=1.0,
            ),
            wakeword_phrases=wakeword_phrases,
            wakeword_stt_phrases=wakeword_stt_phrases,
            wakeword_sample_ms=int(get_value("TWINR_WAKEWORD_SAMPLE_MS", "1800") or "1800"),
            wakeword_presence_grace_s=_parse_float(
                get_value("TWINR_WAKEWORD_PRESENCE_GRACE_S"),
                15.0 * 60.0,
            ),
            wakeword_motion_grace_s=_parse_float(
                get_value("TWINR_WAKEWORD_MOTION_GRACE_S"),
                5.0 * 60.0,
            ),
            wakeword_speech_grace_s=_parse_float(
                get_value("TWINR_WAKEWORD_SPEECH_GRACE_S"),
                90.0,
            ),
            wakeword_attempt_cooldown_s=_parse_float(
                get_value("TWINR_WAKEWORD_ATTEMPT_COOLDOWN_S"),
                4.0,
            ),
            wakeword_block_proactive_after_attempt_s=_parse_float(
                get_value("TWINR_WAKEWORD_BLOCK_PROACTIVE_AFTER_ATTEMPT_S"),
                20.0,
            ),
            wakeword_min_active_ratio=_parse_float(
                get_value("TWINR_WAKEWORD_MIN_ACTIVE_RATIO"),
                0.08,
            ),
            wakeword_min_active_chunks=int(
                get_value("TWINR_WAKEWORD_MIN_ACTIVE_CHUNKS", "2") or "2"
            ),
            wakeword_openwakeword_models=wakeword_openwakeword_models,
            wakeword_openwakeword_custom_verifier_models=wakeword_openwakeword_custom_verifier_models,
            wakeword_openwakeword_custom_verifier_threshold=_parse_clamped_float(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_CUSTOM_VERIFIER_THRESHOLD"),
                0.1,
                minimum=0.0,
                maximum=1.0,
            ),
            wakeword_openwakeword_threshold=_parse_clamped_float(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_THRESHOLD"),
                0.5,
                minimum=MIN_SAFE_OPENWAKEWORD_THRESHOLD,
                maximum=1.0,
            ),
            wakeword_openwakeword_vad_threshold=_parse_float(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_VAD_THRESHOLD"),
                0.0,
            ),
            wakeword_openwakeword_patience_frames=int(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_PATIENCE_FRAMES", "1") or "1"
            ),
            wakeword_openwakeword_activation_samples=int(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_ACTIVATION_SAMPLES", "3") or "3"
            ),
            wakeword_openwakeword_deactivation_threshold=_parse_clamped_float(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_DEACTIVATION_THRESHOLD"),
                0.2,
                minimum=0.0,
                maximum=1.0,
            ),
            wakeword_openwakeword_enable_speex=_parse_bool(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_ENABLE_SPEEX"),
                False,
            ),
            wakeword_openwakeword_transcribe_on_detect=_parse_bool(
                get_value("TWINR_WAKEWORD_OPENWAKEWORD_TRANSCRIBE_ON_DETECT"),
                False,
            ),
            wakeword_openwakeword_inference_framework=wakeword_openwakeword_inference_framework,
            wakeword_calibration_profile_path=get_value(
                "TWINR_WAKEWORD_CALIBRATION_PROFILE_PATH",
                str(project_root / "state" / "wakeword_calibration.json"),
            )
            or str(project_root / "state" / "wakeword_calibration.json"),
            wakeword_calibration_recommended_path=get_value(
                "TWINR_WAKEWORD_CALIBRATION_RECOMMENDED_PATH",
                str(project_root / "state" / "wakeword_calibration.recommended.json"),
            )
            or str(project_root / "state" / "wakeword_calibration.recommended.json"),
            proactive_person_returned_absence_s=_parse_float(
                get_value("TWINR_PROACTIVE_PERSON_RETURNED_ABSENCE_S"),
                20.0 * 60.0,
            ),
            proactive_person_returned_recent_motion_s=_parse_float(
                get_value("TWINR_PROACTIVE_PERSON_RETURNED_RECENT_MOTION_S"),
                30.0,
            ),
            proactive_attention_window_s=_parse_float(
                get_value("TWINR_PROACTIVE_ATTENTION_WINDOW_S"),
                6.0,
            ),
            proactive_slumped_quiet_s=_parse_float(
                get_value("TWINR_PROACTIVE_SLUMPED_QUIET_S"),
                20.0,
            ),
            proactive_possible_fall_stillness_s=_parse_float(
                get_value("TWINR_PROACTIVE_POSSIBLE_FALL_STILLNESS_S"),
                10.0,
            ),
            proactive_possible_fall_visibility_loss_hold_s=_parse_float(
                get_value("TWINR_PROACTIVE_POSSIBLE_FALL_VISIBILITY_LOSS_HOLD_S"),
                15.0,
            ),
            proactive_possible_fall_visibility_loss_arming_s=_parse_float(
                get_value("TWINR_PROACTIVE_POSSIBLE_FALL_VISIBILITY_LOSS_ARMING_S"),
                6.0,
            ),
            proactive_possible_fall_slumped_visibility_loss_arming_s=_parse_float(
                get_value("TWINR_PROACTIVE_POSSIBLE_FALL_SLUMPED_VISIBILITY_LOSS_ARMING_S"),
                4.0,
            ),
            proactive_possible_fall_once_per_presence_session=_parse_bool(
                get_value("TWINR_PROACTIVE_POSSIBLE_FALL_ONCE_PER_PRESENCE_SESSION"),
                True,
            ),
            proactive_floor_stillness_s=_parse_float(
                get_value("TWINR_PROACTIVE_FLOOR_STILLNESS_S"),
                20.0,
            ),
            proactive_showing_intent_hold_s=_parse_float(
                get_value("TWINR_PROACTIVE_SHOWING_INTENT_HOLD_S"),
                1.5,
            ),
            proactive_positive_contact_hold_s=_parse_float(
                get_value("TWINR_PROACTIVE_POSITIVE_CONTACT_HOLD_S"),
                1.5,
            ),
            proactive_distress_hold_s=_parse_float(
                get_value("TWINR_PROACTIVE_DISTRESS_HOLD_S"),
                3.0,
            ),
            proactive_fall_transition_window_s=_parse_float(
                get_value("TWINR_PROACTIVE_FALL_TRANSITION_WINDOW_S"),
                8.0,
            ),
            proactive_person_returned_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_PERSON_RETURNED_SCORE_THRESHOLD"),
                0.9,
            ),
            proactive_attention_window_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_ATTENTION_WINDOW_SCORE_THRESHOLD"),
                0.86,
            ),
            proactive_slumped_quiet_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_SLUMPED_QUIET_SCORE_THRESHOLD"),
                0.9,
            ),
            proactive_possible_fall_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_POSSIBLE_FALL_SCORE_THRESHOLD"),
                0.82,
            ),
            proactive_floor_stillness_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_FLOOR_STILLNESS_SCORE_THRESHOLD"),
                0.9,
            ),
            proactive_showing_intent_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD"),
                0.84,
            ),
            proactive_positive_contact_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_POSITIVE_CONTACT_SCORE_THRESHOLD"),
                0.84,
            ),
            proactive_distress_possible_score_threshold=_parse_float(
                get_value("TWINR_PROACTIVE_DISTRESS_POSSIBLE_SCORE_THRESHOLD"),
                0.85,
            ),
            proactive_governor_enabled=_parse_bool(
                get_value("TWINR_PROACTIVE_GOVERNOR_ENABLED"),
                True,
            ),
            proactive_governor_active_reservation_ttl_s=_parse_float(
                get_value("TWINR_PROACTIVE_GOVERNOR_ACTIVE_RESERVATION_TTL_S"),
                45.0,
            ),
            proactive_governor_global_prompt_cooldown_s=_parse_float(
                get_value("TWINR_PROACTIVE_GOVERNOR_GLOBAL_PROMPT_COOLDOWN_S"),
                120.0,
            ),
            proactive_governor_window_s=_parse_float(
                get_value("TWINR_PROACTIVE_GOVERNOR_WINDOW_S"),
                20.0 * 60.0,
            ),
            proactive_governor_window_prompt_limit=int(
                get_value("TWINR_PROACTIVE_GOVERNOR_WINDOW_PROMPT_LIMIT", "4") or "4"
            ),
            proactive_governor_presence_session_prompt_limit=int(
                get_value("TWINR_PROACTIVE_GOVERNOR_PRESENCE_SESSION_PROMPT_LIMIT", "2") or "2"
            ),
            proactive_governor_source_repeat_cooldown_s=_parse_float(
                get_value("TWINR_PROACTIVE_GOVERNOR_SOURCE_REPEAT_COOLDOWN_S"),
                10.0 * 60.0,
            ),
            proactive_governor_history_limit=int(
                get_value("TWINR_PROACTIVE_GOVERNOR_HISTORY_LIMIT", "128") or "128"
            ),
            proactive_visual_first_audio_global_cooldown_s=_parse_float(
                get_value("TWINR_PROACTIVE_VISUAL_FIRST_AUDIO_GLOBAL_COOLDOWN_S"),
                5.0 * 60.0,
            ),
            proactive_visual_first_audio_source_repeat_cooldown_s=_parse_float(
                get_value("TWINR_PROACTIVE_VISUAL_FIRST_AUDIO_SOURCE_REPEAT_COOLDOWN_S"),
                15.0 * 60.0,
            ),
            proactive_visual_first_cue_hold_s=_parse_float(
                get_value("TWINR_PROACTIVE_VISUAL_FIRST_CUE_HOLD_S"),
                45.0,
            ),
            proactive_quiet_hours_visual_only_enabled=_parse_bool(
                get_value("TWINR_PROACTIVE_QUIET_HOURS_VISUAL_ONLY_ENABLED"),
                True,
            ),
            proactive_quiet_hours_start_local=get_value(
                "TWINR_PROACTIVE_QUIET_HOURS_START_LOCAL",
                "21:00",
            )
            or "21:00",
            proactive_quiet_hours_end_local=get_value(
                "TWINR_PROACTIVE_QUIET_HOURS_END_LOCAL",
                "07:00",
            )
            or "07:00",
            web_host=get_value("TWINR_WEB_HOST", "0.0.0.0") or "0.0.0.0",
            web_port=int(get_value("TWINR_WEB_PORT", "1337") or "1337"),
            runtime_state_path=get_value(
                "TWINR_RUNTIME_STATE_PATH",
                str(project_root / "state" / "runtime-state.json"),
            )
            or str(project_root / "state" / "runtime-state.json"),
            memory_markdown_path=get_value(
                "TWINR_MEMORY_MARKDOWN_PATH",
                str(project_root / "state" / "MEMORY.md"),
            )
            or str(project_root / "state" / "MEMORY.md"),
            reminder_store_path=get_value(
                "TWINR_REMINDER_STORE_PATH",
                str(project_root / "state" / "reminders.json"),
            )
            or str(project_root / "state" / "reminders.json"),
            automation_store_path=get_value(
                "TWINR_AUTOMATION_STORE_PATH",
                str(project_root / "state" / "automations.json"),
            )
            or str(project_root / "state" / "automations.json"),
            voice_profile_store_path=get_value(
                "TWINR_VOICE_PROFILE_STORE_PATH",
                str(project_root / "state" / "voice_profile.json"),
            )
            or str(project_root / "state" / "voice_profile.json"),
            adaptive_timing_enabled=_parse_bool(get_value("TWINR_ADAPTIVE_TIMING_ENABLED"), True),
            adaptive_timing_store_path=get_value(
                "TWINR_ADAPTIVE_TIMING_STORE_PATH",
                str(project_root / "state" / "adaptive_timing.json"),
            )
            or str(project_root / "state" / "adaptive_timing.json"),
            adaptive_timing_pause_grace_ms=int(
                get_value("TWINR_ADAPTIVE_TIMING_PAUSE_GRACE_MS", "450") or "450"
            ),
            long_term_memory_enabled=_parse_bool(get_value("TWINR_LONG_TERM_MEMORY_ENABLED"), False),
            long_term_memory_backend=get_value("TWINR_LONG_TERM_MEMORY_BACKEND", "chonkydb") or "chonkydb",
            long_term_memory_mode=(
                get_value("TWINR_LONG_TERM_MEMORY_MODE", "local_first") or "local_first"
            ).strip().lower(),
            long_term_memory_remote_required=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED"),
                False,
            ),
            long_term_memory_remote_namespace=get_value("TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE") or None,
            long_term_memory_path=get_value(
                "TWINR_LONG_TERM_MEMORY_PATH",
                str(project_root / "state" / "chonkydb"),
            )
            or str(project_root / "state" / "chonkydb"),
            long_term_memory_background_store_turns=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS"),
                True,
            ),
            long_term_memory_write_queue_size=int(
                get_value("TWINR_LONG_TERM_MEMORY_WRITE_QUEUE_SIZE", "32") or "32"
            ),
            long_term_memory_recall_limit=int(
                get_value("TWINR_LONG_TERM_MEMORY_RECALL_LIMIT", "3") or "3"
            ),
            long_term_memory_query_rewrite_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_QUERY_REWRITE_ENABLED"),
                True,
            ),
            long_term_memory_remote_read_timeout_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_READ_TIMEOUT_S"),
                8.0,
            ),
            long_term_memory_remote_write_timeout_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WRITE_TIMEOUT_S"),
                15.0,
            ),
            long_term_memory_remote_keepalive_interval_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_KEEPALIVE_INTERVAL_S"),
                5.0,
                minimum=0.1,
            ),
            long_term_memory_remote_runtime_check_mode=(
                get_value(
                    "TWINR_LONG_TERM_MEMORY_REMOTE_RUNTIME_CHECK_MODE",
                    default_remote_runtime_check_mode,
                )
                or default_remote_runtime_check_mode
            ).strip().lower(),
            long_term_memory_remote_watchdog_startup_wait_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_STARTUP_WAIT_S"),
                30.0,
                minimum=0.0,
                maximum=300.0,
            ),
            long_term_memory_remote_watchdog_interval_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_INTERVAL_S"),
                1.0,
                minimum=0.1,
            ),
            long_term_memory_remote_watchdog_history_limit=max(
                1,
                int(get_value("TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_HISTORY_LIMIT", "3600") or "3600"),
            ),
            long_term_memory_remote_max_content_chars=int(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_MAX_CONTENT_CHARS", "2000000") or "2000000"
            ),
            long_term_memory_remote_shard_max_content_chars=int(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_SHARD_MAX_CONTENT_CHARS", "1000000") or "1000000"
            ),
            long_term_memory_remote_retry_attempts=int(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_RETRY_ATTEMPTS", "3") or "3"
            ),
            long_term_memory_remote_retry_backoff_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_RETRY_BACKOFF_S"),
                1.0,
            ),
            long_term_memory_remote_flush_timeout_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_FLUSH_TIMEOUT_S"),
                60.0,
            ),
            long_term_memory_remote_read_cache_ttl_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_READ_CACHE_TTL_S"),
                0.0,
                minimum=0.0,
            ),
            long_term_memory_turn_extractor_model=(
                get_value("TWINR_LONG_TERM_MEMORY_TURN_EXTRACTOR_MODEL") or None
            ),
            long_term_memory_turn_extractor_max_output_tokens=int(
                get_value("TWINR_LONG_TERM_MEMORY_TURN_EXTRACTOR_MAX_OUTPUT_TOKENS", "2200") or "2200"
            ),
            long_term_memory_midterm_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_MIDTERM_ENABLED"),
                True,
            ),
            long_term_memory_midterm_limit=int(
                get_value("TWINR_LONG_TERM_MEMORY_MIDTERM_LIMIT", "4") or "4"
            ),
            long_term_memory_reflection_window_size=int(
                get_value("TWINR_LONG_TERM_MEMORY_REFLECTION_WINDOW_SIZE", "18") or "18"
            ),
            long_term_memory_reflection_compiler_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_ENABLED"),
                True,
            ),
            long_term_memory_reflection_compiler_model=(
                get_value("TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_MODEL") or None
            ),
            long_term_memory_reflection_compiler_max_output_tokens=int(
                get_value("TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_MAX_OUTPUT_TOKENS", "900") or "900"
            ),
            long_term_memory_subtext_compiler_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_ENABLED"),
                True,
            ),
            long_term_memory_subtext_compiler_model=(
                get_value("TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_MODEL") or None
            ),
            long_term_memory_subtext_compiler_max_output_tokens=int(
                get_value("TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_MAX_OUTPUT_TOKENS", "520") or "520"
            ),
            long_term_memory_proactive_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_ENABLED"),
                False,
            ),
            long_term_memory_proactive_poll_interval_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_POLL_INTERVAL_S"),
                30.0,
            ),
            long_term_memory_proactive_min_confidence=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_MIN_CONFIDENCE"),
                0.72,
            ),
            long_term_memory_proactive_repeat_cooldown_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_REPEAT_COOLDOWN_S"),
                6.0 * 60.0 * 60.0,
            ),
            long_term_memory_proactive_skip_cooldown_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_SKIP_COOLDOWN_S"),
                30.0 * 60.0,
            ),
            long_term_memory_proactive_reservation_ttl_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_RESERVATION_TTL_S"),
                90.0,
            ),
            long_term_memory_proactive_allow_sensitive=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_ALLOW_SENSITIVE"),
                False,
            ),
            long_term_memory_proactive_history_limit=int(
                get_value("TWINR_LONG_TERM_MEMORY_PROACTIVE_HISTORY_LIMIT", "128") or "128"
            ),
            long_term_memory_sensor_memory_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_SENSOR_MEMORY_ENABLED"),
                False,
            ),
            long_term_memory_sensor_baseline_days=int(
                get_value("TWINR_LONG_TERM_MEMORY_SENSOR_BASELINE_DAYS", "21") or "21"
            ),
            long_term_memory_sensor_min_days_observed=int(
                get_value("TWINR_LONG_TERM_MEMORY_SENSOR_MIN_DAYS_OBSERVED", "6") or "6"
            ),
            long_term_memory_sensor_min_routine_ratio=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_SENSOR_MIN_ROUTINE_RATIO"),
                0.55,
            ),
            long_term_memory_sensor_deviation_min_delta=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_SENSOR_DEVIATION_MIN_DELTA"),
                0.45,
            ),
            long_term_memory_retention_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_RETENTION_ENABLED"),
                True,
            ),
            long_term_memory_retention_mode=(
                get_value("TWINR_LONG_TERM_MEMORY_RETENTION_MODE", "conservative") or "conservative"
            ).strip().lower(),
            long_term_memory_retention_run_interval_s=_parse_float(
                get_value("TWINR_LONG_TERM_MEMORY_RETENTION_RUN_INTERVAL_S"),
                300.0,
            ),
            long_term_memory_archive_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_ARCHIVE_ENABLED"),
                True,
            ),
            long_term_memory_migration_enabled=_parse_bool(
                get_value("TWINR_LONG_TERM_MEMORY_MIGRATION_ENABLED"),
                True,
            ),
            long_term_memory_migration_batch_size=int(
                get_value("TWINR_LONG_TERM_MEMORY_MIGRATION_BATCH_SIZE", "64") or "64"
            ),
            long_term_memory_remote_bulk_request_max_bytes=int(
                get_value("TWINR_LONG_TERM_MEMORY_REMOTE_BULK_REQUEST_MAX_BYTES", str(512 * 1024))
                or str(512 * 1024)
            ),
            chonkydb_base_url=(
                get_value("TWINR_CHONKYDB_BASE_URL")
                or get_value("CCODEX_MEMORY_BASE_URL")
            ),
            chonkydb_api_key=(
                get_value("TWINR_CHONKYDB_API_KEY")
                or get_value("CCODEX_MEMORY_API_KEY")
            ),
            chonkydb_api_key_header=get_value("TWINR_CHONKYDB_API_KEY_HEADER", "x-api-key") or "x-api-key",
            chonkydb_allow_bearer_auth=_parse_bool(get_value("TWINR_CHONKYDB_ALLOW_BEARER_AUTH"), False),
            chonkydb_timeout_s=_parse_float(get_value("TWINR_CHONKYDB_TIMEOUT_S"), 20.0),
            chonkydb_max_response_bytes=int(
                get_value("TWINR_CHONKYDB_MAX_RESPONSE_BYTES", str(64 * 1024 * 1024)) or str(64 * 1024 * 1024)
            ),
            restore_runtime_state_on_startup=_parse_bool(
                get_value("TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP"),
                False,
            ),
            reminder_poll_interval_s=_parse_float(get_value("TWINR_REMINDER_POLL_INTERVAL_S"), 1.0),
            reminder_retry_delay_s=_parse_float(get_value("TWINR_REMINDER_RETRY_DELAY_S"), 90.0),
            reminder_max_entries=int(get_value("TWINR_REMINDER_MAX_ENTRIES", "48") or "48"),
            automation_poll_interval_s=_parse_float(get_value("TWINR_AUTOMATION_POLL_INTERVAL_S"), 5.0),
            automation_max_entries=int(get_value("TWINR_AUTOMATION_MAX_ENTRIES", "96") or "96"),
            voice_profile_min_sample_ms=int(get_value("TWINR_VOICE_PROFILE_MIN_SAMPLE_MS", "1200") or "1200"),
            voice_profile_likely_threshold=_parse_float(
                get_value("TWINR_VOICE_PROFILE_LIKELY_THRESHOLD"),
                0.72,
            ),
            voice_profile_uncertain_threshold=_parse_float(
                get_value("TWINR_VOICE_PROFILE_UNCERTAIN_THRESHOLD"),
                0.55,
            ),
            voice_profile_max_samples=int(get_value("TWINR_VOICE_PROFILE_MAX_SAMPLES", "6") or "6"),
            speech_pause_ms=int(get_value("TWINR_SPEECH_PAUSE_MS", "1200") or "1200"),
            memory_max_turns=int(get_value("TWINR_MEMORY_MAX_TURNS", "20") or "20"),
            memory_keep_recent=int(get_value("TWINR_MEMORY_KEEP_RECENT", "10") or "10"),
            gpio_chip=get_value("TWINR_GPIO_CHIP", "gpiochip0") or "gpiochip0",
            green_button_gpio=_parse_optional_int(get_value("TWINR_GREEN_BUTTON_GPIO")),
            yellow_button_gpio=_parse_optional_int(get_value("TWINR_YELLOW_BUTTON_GPIO")),
            pir_motion_gpio=_parse_optional_int(get_value("TWINR_PIR_MOTION_GPIO")),
            pir_active_high=_parse_bool(get_value("TWINR_PIR_ACTIVE_HIGH"), True),
            pir_bias=(get_value("TWINR_PIR_BIAS", "pull-down") or "pull-down").strip().lower(),
            pir_debounce_ms=int(get_value("TWINR_PIR_DEBOUNCE_MS", "120") or "120"),
            button_active_low=_parse_bool(get_value("TWINR_BUTTON_ACTIVE_LOW"), True),
            button_bias=(get_value("TWINR_BUTTON_BIAS", "pull-up") or "pull-up").strip().lower(),
            button_debounce_ms=int(get_value("TWINR_BUTTON_DEBOUNCE_MS", "80") or "80"),
            button_probe_lines=_parse_csv_ints(
                get_value("TWINR_BUTTON_PROBE_LINES"),
                DEFAULT_BUTTON_PROBE_LINES,
            ),
            display_driver=get_value("TWINR_DISPLAY_DRIVER", "hdmi_fbdev") or "hdmi_fbdev",
            display_companion_enabled=_parse_optional_bool(
                get_value("TWINR_DISPLAY_COMPANION_ENABLED")
            ),
            display_fb_path=get_value("TWINR_DISPLAY_FB_PATH", "/dev/fb0") or "/dev/fb0",
            display_wayland_display=get_value("TWINR_DISPLAY_WAYLAND_DISPLAY", "wayland-0") or "wayland-0",
            display_wayland_runtime_dir=get_value("TWINR_DISPLAY_WAYLAND_RUNTIME_DIR"),
            display_face_cue_path=get_value(
                "TWINR_DISPLAY_FACE_CUE_PATH",
                "artifacts/stores/ops/display_face_cue.json",
            )
            or "artifacts/stores/ops/display_face_cue.json",
            display_face_cue_ttl_s=_parse_float(get_value("TWINR_DISPLAY_FACE_CUE_TTL_S"), 4.0, minimum=0.1),
            display_attention_refresh_interval_s=_parse_float(
                get_value("TWINR_DISPLAY_ATTENTION_REFRESH_INTERVAL_S"),
                1.25,
                minimum=0.0,
            ),
            display_presentation_path=get_value(
                "TWINR_DISPLAY_PRESENTATION_PATH",
                "artifacts/stores/ops/display_presentation.json",
            )
            or "artifacts/stores/ops/display_presentation.json",
            display_presentation_ttl_s=_parse_float(
                get_value("TWINR_DISPLAY_PRESENTATION_TTL_S"),
                20.0,
                minimum=0.1,
            ),
            display_vendor_dir=get_value("TWINR_DISPLAY_VENDOR_DIR", "state/display/vendor")
            or "state/display/vendor",
            display_spi_bus=int(get_value("TWINR_DISPLAY_SPI_BUS", "0") or "0"),
            display_spi_device=int(get_value("TWINR_DISPLAY_SPI_DEVICE", "0") or "0"),
            display_cs_gpio=int(get_value("TWINR_DISPLAY_CS_GPIO", "8") or "8"),
            display_dc_gpio=int(get_value("TWINR_DISPLAY_DC_GPIO", "25") or "25"),
            display_reset_gpio=int(get_value("TWINR_DISPLAY_RESET_GPIO", "17") or "17"),
            display_busy_gpio=int(get_value("TWINR_DISPLAY_BUSY_GPIO", "24") or "24"),
            display_width=int(get_value("TWINR_DISPLAY_WIDTH", "400") or "400"),
            display_height=int(get_value("TWINR_DISPLAY_HEIGHT", "300") or "300"),
            display_rotation_degrees=int(get_value("TWINR_DISPLAY_ROTATION_DEGREES", "270") or "270"),
            display_full_refresh_interval=int(get_value("TWINR_DISPLAY_FULL_REFRESH_INTERVAL", "0") or "0"),
            display_busy_timeout_s=_parse_float(get_value("TWINR_DISPLAY_BUSY_TIMEOUT_S"), 20.0, minimum=0.1),
            display_runtime_trace_enabled=_parse_bool(get_value("TWINR_DISPLAY_RUNTIME_TRACE_ENABLED"), False),
            display_poll_interval_s=_parse_float(get_value("TWINR_DISPLAY_POLL_INTERVAL_S"), 0.5),
            display_layout=get_value("TWINR_DISPLAY_LAYOUT", "default") or "default",
            display_news_ticker_enabled=_parse_bool(get_value("TWINR_DISPLAY_NEWS_TICKER_ENABLED"), False),
            display_news_ticker_feed_urls=_parse_csv_strings(
                get_value("TWINR_DISPLAY_NEWS_TICKER_FEED_URLS"),
                (),
            ),
            display_news_ticker_store_path=get_value(
                "TWINR_DISPLAY_NEWS_TICKER_STORE_PATH",
                "artifacts/stores/ops/display_news_ticker.json",
            )
            or "artifacts/stores/ops/display_news_ticker.json",
            display_news_ticker_refresh_interval_s=_parse_float(
                get_value("TWINR_DISPLAY_NEWS_TICKER_REFRESH_INTERVAL_S"),
                900.0,
                minimum=30.0,
            ),
            display_news_ticker_rotation_interval_s=_parse_float(
                get_value("TWINR_DISPLAY_NEWS_TICKER_ROTATION_INTERVAL_S"),
                12.0,
                minimum=4.0,
            ),
            display_news_ticker_max_items=int(get_value("TWINR_DISPLAY_NEWS_TICKER_MAX_ITEMS", "12") or "12"),
            display_news_ticker_timeout_s=_parse_float(
                get_value("TWINR_DISPLAY_NEWS_TICKER_TIMEOUT_S"),
                4.0,
                minimum=0.5,
            ),
            printer_queue=get_value("TWINR_PRINTER_QUEUE", "Thermal_GP58") or "Thermal_GP58",
            printer_device_uri=get_value("TWINR_PRINTER_DEVICE_URI"),
            printer_header_text=get_value("TWINR_PRINTER_HEADER_TEXT", "TWINR.com") or "TWINR.com",
            printer_feed_lines=int(get_value("TWINR_PRINTER_FEED_LINES", "3") or "3"),
            printer_line_width=int(get_value("TWINR_PRINTER_LINE_WIDTH", "30") or "30"),
            print_button_cooldown_s=_parse_float(get_value("TWINR_PRINT_BUTTON_COOLDOWN_S"), 2.0),
            print_max_lines=int(get_value("TWINR_PRINT_MAX_LINES", "8") or "8"),
            print_max_chars=int(get_value("TWINR_PRINT_MAX_CHARS", "320") or "320"),
            print_context_turns=int(get_value("TWINR_PRINT_CONTEXT_TURNS", "6") or "6"),
        )
