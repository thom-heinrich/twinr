"""Load provider selection, model, and API transport settings."""

from __future__ import annotations

from .context import ConfigLoadContext
from .constants import (
    DEFAULT_OPENAI_MAIN_MODEL,
)
from .parsing import (
    _parse_bool,
    _parse_float,
    _parse_optional_bool,
)


def load_provider_config(context: ConfigLoadContext) -> dict[str, object]:
    """Return the config fields owned by this loading domain."""

    get_value = context.get_value
    project_root = context.project_root

    return {
        "openai_api_key": get_value("OPENAI_API_KEY"),
        "openai_project_id": get_value("OPENAI_PROJ_ID"),
        "openai_send_project_header": _parse_optional_bool(
            get_value("OPENAI_SEND_PROJECT_HEADER")
        ),
        "stt_provider": (get_value("TWINR_STT_PROVIDER", "openai") or "openai")
        .strip()
        .lower(),
        "llm_provider": (get_value("TWINR_LLM_PROVIDER", "openai") or "openai")
        .strip()
        .lower(),
        "tts_provider": (get_value("TWINR_TTS_PROVIDER", "openai") or "openai")
        .strip()
        .lower(),
        "project_root": str(project_root),
        "personality_dir": get_value("TWINR_PERSONALITY_DIR", "personality")
        or "personality",
        "user_display_name": get_value("TWINR_USER_DISPLAY_NAME"),
        "default_model": get_value("OPENAI_MODEL", DEFAULT_OPENAI_MAIN_MODEL)
        or DEFAULT_OPENAI_MAIN_MODEL,
        "openai_reasoning_effort": get_value("OPENAI_REASONING_EFFORT", "medium")
        or "medium",
        "openai_prompt_cache_enabled": _parse_bool(
            get_value("OPENAI_PROMPT_CACHE_ENABLED"), True
        ),
        "openai_prompt_cache_retention": get_value("OPENAI_PROMPT_CACHE_RETENTION"),
        "openai_stt_model": get_value("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
        or "gpt-4o-mini-transcribe",
        "openai_tts_model": get_value("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        or "gpt-4o-mini-tts",
        "openai_tts_voice": get_value("OPENAI_TTS_VOICE", "marin") or "marin",
        "openai_tts_speed": _parse_float(get_value("OPENAI_TTS_SPEED"), 1.0),
        "openai_tts_format": get_value("OPENAI_TTS_FORMAT", "wav") or "wav",
        "openai_tts_instructions": get_value("OPENAI_TTS_INSTRUCTIONS"),
        "deepgram_api_key": get_value("DEEPGRAM_API_KEY"),
        "deepgram_base_url": get_value(
            "DEEPGRAM_BASE_URL", "https://api.deepgram.com/v1"
        )
        or "https://api.deepgram.com/v1",
        "deepgram_stt_model": get_value("DEEPGRAM_STT_MODEL", "nova-3") or "nova-3",
        "deepgram_stt_language": get_value("DEEPGRAM_STT_LANGUAGE", "de") or "de",
        "deepgram_stt_smart_format": _parse_bool(
            get_value("DEEPGRAM_STT_SMART_FORMAT"), True
        ),
        "deepgram_streaming_interim_results": _parse_bool(
            get_value("DEEPGRAM_STREAMING_INTERIM_RESULTS"), True
        ),
        "deepgram_streaming_endpointing_ms": int(
            get_value("DEEPGRAM_STREAMING_ENDPOINTING_MS", "400") or "400"
        ),
        "deepgram_streaming_utterance_end_ms": int(
            get_value("DEEPGRAM_STREAMING_UTTERANCE_END_MS", "1000") or "1000"
        ),
        "deepgram_streaming_stop_on_utterance_end": _parse_bool(
            get_value("DEEPGRAM_STREAMING_STOP_ON_UTTERANCE_END"), True
        ),
        "deepgram_streaming_finalize_timeout_s": _parse_float(
            get_value("DEEPGRAM_STREAMING_FINALIZE_TIMEOUT_S"), 4.0
        ),
        "deepgram_timeout_s": _parse_float(get_value("DEEPGRAM_TIMEOUT_S"), 30.0),
        "groq_api_key": get_value("GROQ_API_KEY"),
        "groq_base_url": get_value("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        or "https://api.groq.com/openai/v1",
        "groq_model": get_value("GROQ_MODEL", "llama-3.3-70b-versatile")
        or "llama-3.3-70b-versatile",
        "groq_timeout_s": _parse_float(get_value("GROQ_TIMEOUT_S"), 45.0),
        "openai_realtime_model": get_value("OPENAI_REALTIME_MODEL", "gpt-realtime-1.5")
        or "gpt-realtime-1.5",
        "openai_realtime_voice": get_value("OPENAI_REALTIME_VOICE", "sage") or "sage",
        "openai_realtime_speed": _parse_float(get_value("OPENAI_REALTIME_SPEED"), 1.0),
        "openai_realtime_instructions": get_value("OPENAI_REALTIME_INSTRUCTIONS"),
        "openai_realtime_transcription_model": get_value(
            "OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"
        )
        or "gpt-4o-mini-transcribe",
        "openai_realtime_language": get_value("OPENAI_REALTIME_LANGUAGE", "de") or "de",
        "openai_realtime_input_sample_rate": int(
            get_value("OPENAI_REALTIME_INPUT_SAMPLE_RATE", "24000") or "24000"
        ),
    }
