"""Define the immutable ``TwinrConfig`` surface and lightweight helpers.

Purpose and boundaries:
- Own the public dataclass shape and helper properties for runtime config.
- Delegate env loading and post-init normalization to focused sibling modules.
- Preserve the stable ``TwinrConfig`` API expected across the Twinr codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DEFAULT_BUTTON_PROBE_LINES,
    SUPPORTED_DISPLAY_DRIVERS,
    GPIO_DISPLAY_DRIVERS,
    DEFAULT_OPENAI_MAIN_MODEL,
    DEFAULT_VOICE_ACTIVATION_PHRASES,
)
from .loading import load_twinr_config
from .normalization import normalize_twinr_config


@dataclass(frozen=True, slots=True)
class TwinrConfig:
    """Store the immutable runtime settings snapshot for the base agent.

    The dataclass groups provider selection, streaming and voice-activation tuning,
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
    default_model: str = DEFAULT_OPENAI_MAIN_MODEL
    openai_reasoning_effort: str = "medium"
    openai_prompt_cache_enabled: bool = True
    openai_prompt_cache_retention: str | None = None
    openai_stt_model: str = "gpt-4o-mini-transcribe"
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
    groq_request_timeout_seconds: float | None = None
    groq_max_retries: int | None = None
    groq_connect_timeout_s: float | None = None
    groq_write_timeout_s: float | None = None
    groq_pool_timeout_s: float | None = None
    groq_max_connections: int | None = None
    groq_max_keepalive_connections: int | None = None
    groq_keepalive_expiry_s: float | None = None
    groq_http2: bool | None = None
    groq_trust_env: bool | None = None
    groq_follow_redirects: bool | None = None
    groq_sdk_backend: str | None = None
    groq_tool_continuation_ttl_seconds: float | None = None
    groq_tool_max_continuations: int | None = None
    groq_max_tool_result_chars: int | None = None
    groq_service_tier: str | None = None
    groq_reasoning_format: str = ""
    groq_reasoning_effort: str = ""
    groq_text_search_model: str | None = None
    groq_vision_model: str | None = None
    groq_allow_search_fallback: bool = False
    groq_allow_vision_fallback: bool = False
    openai_realtime_model: str = "gpt-realtime-1.5"
    openai_realtime_voice: str = "sage"
    openai_realtime_speed: float = 1.0
    openai_realtime_instructions: str | None = None
    openai_realtime_transcription_model: str = "gpt-4o-mini-transcribe"
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
    turn_controller_interrupt_min_transcribe_interval_ms: int = 0
    streaming_early_transcript_enabled: bool = True
    streaming_early_transcript_min_chars: int = 10
    streaming_early_transcript_wait_ms: int = 250
    streaming_transcript_verifier_enabled: bool = True
    streaming_transcript_verifier_lazy_init: bool = True
    streaming_transcript_verifier_model: str = "gpt-4o-mini-transcribe"
    streaming_transcript_verifier_max_words: int = 6
    streaming_transcript_verifier_max_chars: int = 32
    streaming_transcript_verifier_min_confidence: float = 0.92
    streaming_transcript_verifier_max_capture_ms: int = 6500
    streaming_dual_lane_enabled: bool = True
    streaming_first_word_enabled: bool = True
    streaming_first_word_model: str = ""
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
    streaming_search_final_lane_watchdog_timeout_ms: int = 6000
    streaming_search_final_lane_hard_timeout_ms: int = 30000
    realtime_sensitive_tools_require_identity: bool = True
    realtime_sensitive_tools_start_authorized: bool = False
    realtime_sensitive_tool_fragments: tuple[str, ...] = ()
    realtime_sensitive_tool_names: tuple[str, ...] = ()
    streaming_supervisor_model: str = ""
    streaming_supervisor_reasoning_effort: str = "low"
    streaming_supervisor_context_turns: int = 4
    streaming_supervisor_max_output_tokens: int = 80
    streaming_supervisor_prefetch_enabled: bool = True
    streaming_supervisor_prefetch_min_chars: int = 8
    streaming_supervisor_prefetch_wait_ms: int = 80
    streaming_specialist_model: str | None = None
    streaming_specialist_reasoning_effort: str | None = "low"
    local_semantic_router_mode: str = "off"
    local_semantic_router_model_dir: str | None = None
    local_semantic_router_user_intent_model_dir: str | None = None
    local_semantic_router_trace: bool = True
    local_semantic_router_warmup_enabled: bool = False
    local_semantic_router_warmup_probe: str | None = None
    conversation_follow_up_enabled: bool = False
    conversation_follow_up_after_proactive_enabled: bool = False
    conversation_closure_guard_enabled: bool = True
    conversation_closure_model: str = ""
    conversation_closure_reasoning_effort: str = ""
    conversation_closure_context_turns: int = 4
    conversation_closure_instructions_file: str = "CONVERSATION_CLOSURE.md"
    conversation_closure_max_output_tokens: int = 32
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
    search_feedback_pause_ms: int = 1100
    search_feedback_volume: float = 0.09
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
    openai_search_model: str = ""
    openai_web_search_context_size: str = "medium"
    openai_search_max_output_tokens: int = 1024
    openai_search_retry_max_output_tokens: int = 1536
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
    orchestrator_allow_insecure_ws: bool = False
    orchestrator_shared_secret: str | None = None
    voice_orchestrator_enabled: bool = False
    voice_orchestrator_ws_url: str = ""
    voice_orchestrator_allow_insecure_ws: bool = False
    voice_orchestrator_shared_secret: str | None = None
    voice_orchestrator_audio_device: str | None = None
    voice_activation_phrases: tuple[str, ...] = DEFAULT_VOICE_ACTIVATION_PHRASES
    voice_orchestrator_history_ms: int = 4000
    voice_orchestrator_wake_candidate_window_ms: int = 2200
    voice_orchestrator_wake_candidate_min_active_ratio: float = 0.0
    voice_orchestrator_wake_candidate_min_transcript_chars: int = 4
    voice_orchestrator_wake_postroll_ms: int = 250
    voice_orchestrator_wake_tail_max_ms: int = 2200
    voice_orchestrator_wake_tail_endpoint_silence_ms: int = 300
    voice_orchestrator_remote_asr_url: str | None = None
    voice_orchestrator_remote_asr_bearer_token: str | None = None
    voice_orchestrator_remote_asr_min_wake_duration_ms: int = 300
    voice_orchestrator_remote_asr_timeout_s: float = 3.0
    voice_orchestrator_remote_asr_tail_timeout_s: float = 1.25
    voice_orchestrator_remote_asr_language: str | None = None
    voice_orchestrator_remote_asr_mode: str = "active_listening"
    voice_orchestrator_remote_asr_retry_attempts: int = 1
    voice_orchestrator_remote_asr_retry_backoff_s: float = 0.35
    voice_orchestrator_intent_stage1_window_bonus_ms: int = 400
    voice_orchestrator_intent_min_wake_duration_relief_ms: int = 100
    voice_orchestrator_intent_follow_up_timeout_bonus_s: float = 1.5
    voice_orchestrator_follow_up_timeout_s: float = 6.0
    voice_orchestrator_follow_up_window_ms: int = 900
    voice_orchestrator_follow_up_min_active_ratio: float = 0.22
    voice_orchestrator_follow_up_min_transcript_chars: int = 4
    voice_orchestrator_barge_in_window_ms: int = 850
    voice_orchestrator_barge_in_min_active_ratio: float = 0.28
    voice_orchestrator_barge_in_min_transcript_chars: int = 4
    voice_orchestrator_candidate_cooldown_s: float = 0.9
    voice_orchestrator_audio_debug_enabled: bool = False
    voice_orchestrator_audio_debug_dir: str | None = None
    voice_orchestrator_audio_debug_max_files: int = 24
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
    camera_host_mode: str = "onboard"
    camera_second_pi_base_url: str | None = None
    camera_device: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_framerate: int = 30
    camera_input_format: str | None = None
    camera_ffmpeg_path: str = "ffmpeg"
    camera_proxy_snapshot_url: str | None = None
    vision_reference_image_path: str | None = None
    portrait_match_enabled: bool = True
    portrait_match_detector_model_path: str = (
        "state/opencv/models/face_detection_yunet_2023mar.onnx"
    )
    portrait_match_recognizer_model_path: str = (
        "state/opencv/models/face_recognition_sface_2021dec.onnx"
    )
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
    proactive_remote_camera_base_url: str | None = None
    proactive_remote_camera_timeout_s: float = 4.0
    proactive_local_camera_detection_network_path: str = (
        "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    )
    proactive_local_camera_pose_network_path: str = (
        "/usr/share/imx500-models/imx500_network_posenet.rpk"
    )
    proactive_local_camera_pose_backend: str = "mediapipe"
    proactive_local_camera_mediapipe_pose_model_path: str = (
        "state/mediapipe/models/pose_landmarker_full.task"
    )
    proactive_local_camera_mediapipe_hand_landmarker_model_path: str = (
        "state/mediapipe/models/hand_landmarker.task"
    )
    proactive_local_camera_mediapipe_gesture_model_path: str = (
        "state/mediapipe/models/gesture_recognizer.task"
    )
    proactive_local_camera_mediapipe_custom_gesture_model_path: str | None = None
    proactive_local_camera_mediapipe_num_hands: int = 2
    proactive_local_camera_sequence_window_s: float = 0.55
    proactive_local_camera_sequence_min_frames: int = 3
    proactive_local_camera_source_device: str = "imx500"
    proactive_local_camera_frame_rate: int = 15
    proactive_local_camera_lock_timeout_s: float = 5.0
    proactive_local_camera_startup_warmup_s: float = 0.8
    proactive_local_camera_metadata_wait_s: float = 0.75
    proactive_local_camera_person_confidence_threshold: float = 0.40
    proactive_local_camera_object_confidence_threshold: float = 0.55
    proactive_local_camera_person_near_area_threshold: float = 0.20
    proactive_local_camera_person_near_height_threshold: float = 0.55
    proactive_local_camera_object_near_area_threshold: float = 0.08
    proactive_local_camera_attention_score_threshold: float = 0.62
    proactive_local_camera_engaged_score_threshold: float = 0.45
    proactive_local_camera_pose_confidence_threshold: float = 0.30
    proactive_local_camera_pose_refresh_s: float = 0.75
    proactive_local_camera_builtin_gesture_min_score: float = 0.35
    proactive_local_camera_custom_gesture_min_score: float = 0.45
    proactive_local_camera_min_hand_detection_confidence: float = 0.35
    proactive_local_camera_min_hand_presence_confidence: float = 0.35
    proactive_local_camera_min_hand_tracking_confidence: float = 0.35
    proactive_local_camera_max_roi_candidates: int = 4
    proactive_local_camera_primary_person_roi_padding: float = 0.18
    proactive_local_camera_primary_person_upper_body_ratio: float = 0.78
    proactive_local_camera_wrist_roi_scale: float = 0.34
    proactive_local_camera_fine_hand_explicit_hold_s: float = 0.45
    proactive_local_camera_fine_hand_explicit_confirm_samples: int = 1
    proactive_local_camera_fine_hand_explicit_min_confidence: float = 0.72
    gesture_wakeup_enabled: bool = True
    gesture_wakeup_trigger: str = "peace_sign"
    gesture_wakeup_cooldown_s: float = 3.0
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
    proactive_governor_presence_session_window_s: float = 0.0
    proactive_governor_presence_grace_s: float = 0.0
    proactive_governor_source_repeat_cooldown_s: float = 10.0 * 60.0
    proactive_governor_history_limit: int = 128
    proactive_visual_first_audio_global_cooldown_s: float = 5.0 * 60.0
    proactive_visual_first_audio_source_repeat_cooldown_s: float = 15.0 * 60.0
    proactive_visual_first_cue_hold_s: float = 45.0
    proactive_quiet_hours_visual_only_enabled: bool = True
    proactive_quiet_hours_start_local: str = "21:00"
    proactive_quiet_hours_end_local: str = "07:00"
    drone_enabled: bool = False
    drone_base_url: str | None = None
    drone_require_manual_arm: bool = True
    drone_mission_timeout_s: float = 45.0
    drone_request_timeout_s: float = 2.0
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
    long_term_memory_fast_topic_enabled: bool = True
    long_term_memory_fast_topic_limit: int = 3
    long_term_memory_fast_topic_timeout_s: float = 0.6
    long_term_memory_query_rewrite_enabled: bool = True
    long_term_memory_remote_read_timeout_s: float = 8.0
    long_term_memory_remote_write_timeout_s: float = 15.0
    long_term_memory_remote_keepalive_interval_s: float = 5.0
    long_term_memory_remote_runtime_check_mode: str = "direct"
    long_term_memory_remote_watchdog_startup_wait_s: float = 30.0
    long_term_memory_remote_watchdog_interval_s: float = 1.0
    long_term_memory_remote_watchdog_probe_timeout_s: float = 15.0
    long_term_memory_remote_watchdog_startup_probe_timeout_s: float = 45.0
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
    long_term_memory_environment_short_baseline_days: int = 14
    long_term_memory_environment_long_baseline_days: int = 56
    long_term_memory_environment_min_baseline_days: int = 7
    long_term_memory_environment_acute_z_threshold: float = 3.0
    long_term_memory_environment_acute_empirical_q: float = 0.01
    long_term_memory_environment_drift_min_sigma: float = 1.5
    long_term_memory_environment_drift_min_days: int = 5
    long_term_memory_environment_regime_accept_days: int = 10
    long_term_memory_environment_min_coverage_ratio: float = 0.8
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
    browser_automation_enabled: bool = False
    browser_automation_workspace_path: str = "browser_automation"
    browser_automation_entry_module: str = "adapter.py"
    smart_home_background_worker_enabled: bool = True
    smart_home_background_idle_sleep_s: float = 1.0
    smart_home_background_retry_delay_s: float = 2.0
    smart_home_background_batch_limit: int = 8
    smart_home_same_room_entity_ids: tuple[str, ...] = ()
    smart_home_same_room_motion_window_s: float = 90.0
    smart_home_same_room_button_window_s: float = 30.0
    smart_home_home_occupancy_window_s: float = 300.0
    smart_home_stream_stale_after_s: float = 120.0
    voice_profile_min_sample_ms: int = 1200
    voice_profile_likely_threshold: float = 0.72
    voice_profile_uncertain_threshold: float = 0.55
    voice_profile_max_samples: int = 6
    voice_familiar_speaker_min_confidence: float = 0.82
    voice_profile_passive_update_enabled: bool = True
    voice_profile_passive_update_min_confidence: float = 0.86
    voice_profile_passive_update_min_duration_ms: int = 2500
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
    respeaker_led_enabled: bool | None = None
    display_fb_path: str = "/dev/fb0"
    display_wayland_display: str = "wayland-0"
    display_wayland_runtime_dir: str | None = None
    display_face_cue_path: str = "artifacts/stores/ops/display_face_cue.json"
    display_face_cue_ttl_s: float = 4.0
    display_emoji_cue_path: str = "artifacts/stores/ops/display_emoji.json"
    display_emoji_cue_ttl_s: float = 6.0
    display_ambient_impulse_path: str = (
        "artifacts/stores/ops/display_ambient_impulse.json"
    )
    display_ambient_impulse_ttl_s: float = 18.0
    display_service_connect_path: str = (
        "artifacts/stores/ops/display_service_connect.json"
    )
    display_ambient_impulses_enabled: bool = True
    display_reserve_generation_enabled: bool = True
    display_reserve_generation_model: str = ""
    display_reserve_generation_reasoning_effort: str = "low"
    display_reserve_generation_timeout_seconds: float = 20.0
    display_reserve_generation_max_output_tokens: int = 900
    display_reserve_generation_batch_size: int = 2
    display_reserve_generation_variants_per_candidate: int = 3
    display_reserve_bus_plan_path: str = (
        "artifacts/stores/ops/display_reserve_bus_plan.json"
    )
    display_reserve_bus_prepared_plan_path: str = (
        "artifacts/stores/ops/display_reserve_bus_plan_prepared.json"
    )
    display_reserve_bus_maintenance_state_path: str = (
        "artifacts/stores/ops/display_reserve_bus_maintenance.json"
    )
    display_reserve_bus_refresh_after_local: str = "05:30"
    display_reserve_bus_nightly_enabled: bool = True
    display_reserve_bus_nightly_after_local: str = "00:30"
    display_reserve_bus_nightly_poll_interval_s: float = 300.0
    display_reserve_bus_candidate_limit: int = 20
    display_reserve_bus_items_per_day: int = 20
    display_reserve_bus_topic_gap: int = 2
    display_reserve_bus_learning_window_days: float = 21.0
    display_reserve_bus_learning_half_life_days: float = 7.0
    display_reserve_bus_reflection_candidate_limit: int = 3
    display_reserve_bus_reflection_max_age_days: float = 14.0
    display_reserve_bus_min_hold_s: float = 240.0
    display_reserve_bus_base_hold_s: float = 480.0
    display_reserve_bus_max_hold_s: float = 720.0
    display_gesture_refresh_interval_s: float = 0.2
    display_attention_refresh_interval_s: float = 0.2
    display_attention_session_focus_hold_s: float = 4.5
    attention_servo_enabled: bool = False
    attention_servo_forensic_trace_enabled: bool = False
    attention_servo_driver: str = "auto"
    attention_servo_control_mode: str = "position"
    attention_servo_maestro_device: str | None = None
    attention_servo_maestro_channel: int | None = None
    attention_servo_peer_base_url: str | None = None
    attention_servo_peer_timeout_s: float = 1.5
    attention_servo_state_path: str = "state/attention_servo_state.json"
    attention_servo_estimated_zero_max_uncertainty_degrees: float = 15.0
    attention_servo_estimated_zero_settle_tolerance_degrees: float = 1.0
    attention_servo_estimated_zero_speed_scale: float = 0.5
    attention_servo_estimated_zero_move_pulse_delta_us: int = 70
    attention_servo_estimated_zero_move_period_s: float = 0.8
    attention_servo_estimated_zero_move_duty_cycle: float = 0.2
    attention_servo_continuous_return_to_zero_after_s: float = 0.0
    attention_servo_gpio: int | None = None
    attention_servo_invert_direction: bool = False
    attention_servo_target_hold_s: float = 1.1
    attention_servo_loss_extrapolation_s: float = 0.8
    attention_servo_loss_extrapolation_gain: float = 0.65
    attention_servo_min_confidence: float = 0.58
    attention_servo_hold_min_confidence: float = 0.58
    attention_servo_deadband: float = 0.045
    attention_servo_min_pulse_width_us: int = 1050
    attention_servo_center_pulse_width_us: int = 1500
    attention_servo_max_pulse_width_us: int = 1950
    attention_servo_max_step_us: int = 45
    attention_servo_target_smoothing_s: float = 0.9
    attention_servo_max_velocity_us_per_s: float = 80.0
    attention_servo_max_acceleration_us_per_s2: float = 220.0
    attention_servo_max_jerk_us_per_s3: float = 900.0
    attention_servo_rest_max_velocity_us_per_s: float = 35.0
    attention_servo_rest_max_acceleration_us_per_s2: float = 120.0
    attention_servo_rest_max_jerk_us_per_s3: float = 450.0
    attention_servo_min_command_delta_us: int = 8
    attention_servo_visible_retarget_tolerance_us: int = 40
    attention_servo_soft_limit_margin_us: int = 70
    attention_servo_idle_release_s: float = 1.0
    attention_servo_settled_release_s: float = 0.0
    attention_servo_follow_exit_only: bool = False
    attention_servo_visible_recenter_interval_s: float = 30.0
    attention_servo_visible_recenter_center_tolerance: float = 0.12
    attention_servo_mechanical_range_degrees: float = 270.0
    attention_servo_exit_follow_max_degrees: float = 60.0
    attention_servo_exit_activation_delay_s: float = 0.75
    attention_servo_exit_settle_hold_s: float = 0.6
    attention_servo_exit_reacquire_center_tolerance: float = 0.08
    attention_servo_exit_visible_edge_threshold: float = 0.62
    attention_servo_exit_visible_box_edge_threshold: float = 0.92
    attention_servo_exit_cooldown_s: float = 30.0
    attention_servo_continuous_max_speed_degrees_per_s: float = 120.0
    attention_servo_continuous_slow_zone_degrees: float = 45.0
    attention_servo_continuous_stop_tolerance_degrees: float = 4.0
    attention_servo_continuous_min_speed_pulse_delta_us: int = 70
    attention_servo_continuous_max_speed_pulse_delta_us: int = 160
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
    display_poll_interval_s: float = 0.12
    display_layout: str = "default"
    display_news_ticker_enabled: bool = False
    display_news_ticker_legacy_feed_urls: tuple[str, ...] = ()
    display_news_ticker_store_path: str = (
        "artifacts/stores/ops/display_news_ticker.json"
    )
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
        """Delegate post-construction normalization to the focused helper."""

        normalize_twinr_config(self)

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
            other_labels = [
                label for label in labels if not label.startswith("Display ")
            ]
            if other_labels:
                for display_label in display_labels:
                    for other_label in other_labels:
                        conflicts.append(
                            f"{display_label} GPIO {line} collides with {other_label} GPIO {line}."
                        )
                continue
            for index, left in enumerate(display_labels):
                for right in display_labels[index + 1 :]:
                    conflicts.append(
                        f"{left} GPIO {line} collides with {right} GPIO {line}."
                    )
        return tuple(conflicts)

    @property
    def pir_enabled(self) -> bool:
        """Return whether PIR motion sensing is configured."""

        return self.pir_motion_gpio is not None

    @property
    def local_timezone_name(self) -> str:
        """Return the configured local timezone name with a stable fallback."""

        return (
            self.openai_web_search_timezone or "Europe/Berlin"
        ).strip() or "Europe/Berlin"

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "TwinrConfig":
        """Build a config snapshot from process env and a dotenv file."""

        return load_twinr_config(cls, env_path)
