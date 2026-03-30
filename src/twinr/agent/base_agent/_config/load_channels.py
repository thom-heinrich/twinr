"""Load orchestrator, live voice gateway, and WhatsApp channel settings."""

from __future__ import annotations

from .context import ConfigLoadContext
from .parsing import (
    _parse_bool,
    _parse_clamped_float,
    _parse_float,
    _parse_optional_text,
)


def load_channels_config(context: ConfigLoadContext) -> dict[str, object]:
    """Return the config fields owned by this loading domain."""

    get_value = context.get_value
    project_root = context.project_root
    voice_activation_phrases = context.voice_activation_phrases

    return {
        "orchestrator_host": get_value("TWINR_ORCHESTRATOR_HOST", "0.0.0.0")
        or "0.0.0.0",
        "orchestrator_port": int(
            get_value("TWINR_ORCHESTRATOR_PORT", "8797") or "8797"
        ),
        "orchestrator_ws_url": get_value(
            "TWINR_ORCHESTRATOR_WS_URL", "ws://127.0.0.1:8797/ws/orchestrator"
        )
        or "ws://127.0.0.1:8797/ws/orchestrator",
        "orchestrator_allow_insecure_ws": _parse_bool(
            (
                get_value("TWINR_ORCHESTRATOR_ALLOW_INSECURE_WS")
                if get_value("TWINR_ORCHESTRATOR_ALLOW_INSECURE_WS") not in (None, "")
                else get_value("TWINR_ALLOW_INSECURE_ORCHESTRATOR_WS")
            ),
            False,
        ),
        "orchestrator_shared_secret": get_value("TWINR_ORCHESTRATOR_SHARED_SECRET")
        or None,
        "voice_orchestrator_enabled": _parse_bool(
            get_value("TWINR_VOICE_ORCHESTRATOR_ENABLED"), False
        ),
        "voice_orchestrator_ws_url": _parse_optional_text(
            get_value("TWINR_VOICE_ORCHESTRATOR_WS_URL")
        )
        or "",
        "voice_orchestrator_allow_insecure_ws": _parse_bool(
            (
                get_value("TWINR_VOICE_ORCHESTRATOR_ALLOW_INSECURE_WS")
                if get_value("TWINR_VOICE_ORCHESTRATOR_ALLOW_INSECURE_WS")
                not in (None, "")
                else get_value("TWINR_ALLOW_INSECURE_VOICE_WS")
            ),
            False,
        ),
        "voice_orchestrator_shared_secret": get_value(
            "TWINR_VOICE_ORCHESTRATOR_SHARED_SECRET"
        )
        or get_value("TWINR_ORCHESTRATOR_SHARED_SECRET")
        or None,
        "voice_orchestrator_audio_device": get_value(
            "TWINR_VOICE_ORCHESTRATOR_AUDIO_DEVICE"
        )
        or None,
        "voice_activation_phrases": voice_activation_phrases,
        "voice_orchestrator_history_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_HISTORY_MS", "4000") or "4000"
        ),
        "voice_orchestrator_wake_candidate_window_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_WAKE_CANDIDATE_WINDOW_MS", "2200")
            or "2200"
        ),
        "voice_orchestrator_wake_candidate_min_active_ratio": _parse_clamped_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_WAKE_CANDIDATE_MIN_ACTIVE_RATIO"),
            0.18,
            minimum=0.0,
            maximum=1.0,
        ),
        "voice_orchestrator_wake_candidate_min_transcript_chars": max(
            1,
            int(
                get_value(
                    "TWINR_VOICE_ORCHESTRATOR_WAKE_CANDIDATE_MIN_TRANSCRIPT_CHARS", "4"
                )
                or "4"
            ),
        ),
        "voice_orchestrator_wake_postroll_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_WAKE_POSTROLL_MS", "250") or "250"
        ),
        "voice_orchestrator_wake_tail_max_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_WAKE_TAIL_MAX_MS", "2200") or "2200"
        ),
        "voice_orchestrator_wake_tail_endpoint_silence_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_WAKE_TAIL_ENDPOINT_SILENCE_MS", "300")
            or "300"
        ),
        "voice_orchestrator_remote_asr_url": _parse_optional_text(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_URL")
        ),
        "voice_orchestrator_remote_asr_bearer_token": _parse_optional_text(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_BEARER_TOKEN")
        ),
        "voice_orchestrator_remote_asr_min_wake_duration_ms": max(
            0,
            int(
                get_value(
                    "TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_MIN_WAKE_DURATION_MS", "300"
                )
                or "300"
            ),
        ),
        "voice_orchestrator_remote_asr_timeout_s": _parse_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_TIMEOUT_S"),
            3.0,
            minimum=0.25,
        ),
        "voice_orchestrator_remote_asr_tail_timeout_s": _parse_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_TAIL_TIMEOUT_S"),
            1.25,
            minimum=0.25,
        ),
        "voice_orchestrator_remote_asr_language": _parse_optional_text(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_LANGUAGE")
        ),
        "voice_orchestrator_remote_asr_mode": _parse_optional_text(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_MODE")
        )
        or "active_listening",
        "voice_orchestrator_remote_asr_retry_attempts": max(
            0,
            int(
                get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_RETRY_ATTEMPTS", "1")
                or "1"
            ),
        ),
        "voice_orchestrator_remote_asr_retry_backoff_s": _parse_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_REMOTE_ASR_RETRY_BACKOFF_S"),
            0.35,
            minimum=0.0,
        ),
        "voice_orchestrator_intent_stage1_window_bonus_ms": max(
            0,
            int(
                get_value(
                    "TWINR_VOICE_ORCHESTRATOR_INTENT_STAGE1_WINDOW_BONUS_MS", "400"
                )
                or "400"
            ),
        ),
        "voice_orchestrator_intent_min_wake_duration_relief_ms": max(
            0,
            int(
                get_value(
                    "TWINR_VOICE_ORCHESTRATOR_INTENT_MIN_WAKE_DURATION_RELIEF_MS", "100"
                )
                or "100"
            ),
        ),
        "voice_orchestrator_intent_follow_up_timeout_bonus_s": _parse_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_INTENT_FOLLOW_UP_TIMEOUT_BONUS_S"),
            1.5,
            minimum=0.0,
        ),
        "voice_orchestrator_follow_up_timeout_s": _parse_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_FOLLOW_UP_TIMEOUT_S"), 6.0, minimum=1.0
        ),
        "voice_orchestrator_follow_up_window_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_FOLLOW_UP_WINDOW_MS", "900") or "900"
        ),
        "voice_orchestrator_follow_up_min_active_ratio": _parse_clamped_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_FOLLOW_UP_MIN_ACTIVE_RATIO"),
            0.22,
            minimum=0.0,
            maximum=1.0,
        ),
        "voice_orchestrator_follow_up_min_transcript_chars": max(
            1,
            int(
                get_value(
                    "TWINR_VOICE_ORCHESTRATOR_FOLLOW_UP_MIN_TRANSCRIPT_CHARS", "4"
                )
                or "4"
            ),
        ),
        "voice_orchestrator_barge_in_window_ms": int(
            get_value("TWINR_VOICE_ORCHESTRATOR_BARGE_IN_WINDOW_MS", "850") or "850"
        ),
        "voice_orchestrator_barge_in_min_active_ratio": _parse_clamped_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_BARGE_IN_MIN_ACTIVE_RATIO"),
            0.28,
            minimum=0.0,
            maximum=1.0,
        ),
        "voice_orchestrator_barge_in_min_transcript_chars": max(
            1,
            int(
                get_value("TWINR_VOICE_ORCHESTRATOR_BARGE_IN_MIN_TRANSCRIPT_CHARS", "4")
                or "4"
            ),
        ),
        "voice_orchestrator_candidate_cooldown_s": _parse_float(
            get_value("TWINR_VOICE_ORCHESTRATOR_CANDIDATE_COOLDOWN_S"), 0.9, minimum=0.1
        ),
        "voice_orchestrator_audio_debug_enabled": _parse_bool(
            get_value("TWINR_VOICE_ORCHESTRATOR_AUDIO_DEBUG_ENABLED"), False
        ),
        "voice_orchestrator_audio_debug_dir": _parse_optional_text(
            get_value("TWINR_VOICE_ORCHESTRATOR_AUDIO_DEBUG_DIR")
        ),
        "voice_orchestrator_audio_debug_max_files": max(
            4,
            int(
                get_value("TWINR_VOICE_ORCHESTRATOR_AUDIO_DEBUG_MAX_FILES", "24")
                or "24"
            ),
        ),
        "whatsapp_node_binary": get_value("TWINR_WHATSAPP_NODE_BINARY", "node")
        or "node",
        "whatsapp_allow_from": get_value("TWINR_WHATSAPP_ALLOW_FROM") or None,
        "whatsapp_auth_dir": get_value(
            "TWINR_WHATSAPP_AUTH_DIR",
            str(project_root / "state" / "channels" / "whatsapp" / "auth"),
        )
        or str(project_root / "state" / "channels" / "whatsapp" / "auth"),
        "whatsapp_worker_root": get_value(
            "TWINR_WHATSAPP_WORKER_ROOT",
            str(project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"),
        )
        or str(project_root / "src" / "twinr" / "channels" / "whatsapp" / "worker"),
        "whatsapp_groups_enabled": _parse_bool(
            get_value("TWINR_WHATSAPP_GROUPS_ENABLED"), False
        ),
        "whatsapp_self_chat_mode": _parse_bool(
            get_value("TWINR_WHATSAPP_SELF_CHAT_MODE"), False
        ),
        "whatsapp_reconnect_base_delay_s": _parse_float(
            get_value("TWINR_WHATSAPP_RECONNECT_BASE_DELAY_S"), 2.0, minimum=0.1
        ),
        "whatsapp_reconnect_max_delay_s": _parse_float(
            get_value("TWINR_WHATSAPP_RECONNECT_MAX_DELAY_S"), 30.0, minimum=0.1
        ),
        "whatsapp_send_timeout_s": _parse_float(
            get_value("TWINR_WHATSAPP_SEND_TIMEOUT_S"), 20.0, minimum=1.0
        ),
        "whatsapp_sent_cache_ttl_s": _parse_float(
            get_value("TWINR_WHATSAPP_SENT_CACHE_TTL_S"), 180.0, minimum=1.0
        ),
        "whatsapp_sent_cache_max_entries": max(
            16, int(get_value("TWINR_WHATSAPP_SENT_CACHE_MAX_ENTRIES", "256") or "256")
        ),
    }
