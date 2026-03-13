from __future__ import annotations

from dataclasses import dataclass
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


def _read_dotenv(path: Path) -> dict[str, str]:
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
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None
    return _parse_bool(value, False)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _parse_float(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    return float(value)


def _parse_csv_ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not value.strip():
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class TwinrConfig:
    openai_api_key: str | None = None
    openai_project_id: str | None = None
    openai_send_project_header: bool | None = None
    project_root: str = "."
    personality_dir: str = "personality"
    user_display_name: str | None = None
    default_model: str = "gpt-5.2"
    openai_reasoning_effort: str = "medium"
    openai_stt_model: str = "whisper-1"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "marin"
    openai_tts_format: str = "wav"
    openai_tts_instructions: str | None = None
    openai_realtime_model: str = "gpt-4o-realtime-preview"
    openai_realtime_voice: str = "sage"
    openai_realtime_instructions: str | None = None
    openai_realtime_transcription_model: str = "whisper-1"
    openai_realtime_language: str | None = "de"
    openai_realtime_input_sample_rate: int = 24000
    conversation_follow_up_enabled: bool = False
    conversation_follow_up_timeout_s: float = 4.0
    audio_beep_frequency_hz: int = 1046
    audio_beep_duration_ms: int = 180
    audio_beep_volume: float = 0.8
    audio_beep_settle_ms: int = 120
    search_feedback_tones_enabled: bool = True
    search_feedback_delay_ms: int = 1200
    search_feedback_pause_ms: int = 900
    search_feedback_volume: float = 0.14
    audio_speech_start_chunks: int = 1
    audio_follow_up_speech_start_chunks: int = 4
    audio_follow_up_ignore_ms: int = 300
    openai_enable_web_search: bool = False
    openai_search_model: str = "gpt-5.2-chat-latest"
    openai_web_search_context_size: str = "medium"
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
    camera_device: str = "/dev/video0"
    camera_width: int = 640
    camera_height: int = 480
    camera_framerate: int = 30
    camera_input_format: str | None = None
    camera_ffmpeg_path: str = "ffmpeg"
    vision_reference_image_path: str | None = None
    openai_vision_detail: str = "auto"
    proactive_enabled: bool = False
    proactive_poll_interval_s: float = 4.0
    proactive_capture_interval_s: float = 6.0
    proactive_motion_window_s: float = 20.0
    proactive_low_motion_after_s: float = 12.0
    proactive_audio_enabled: bool = False
    proactive_audio_input_device: str | None = None
    proactive_audio_sample_ms: int = 1000
    proactive_audio_distress_enabled: bool = False
    web_host: str = "0.0.0.0"
    web_port: int = 1337
    runtime_state_path: str = "/tmp/twinr-runtime-state.json"
    memory_markdown_path: str = "state/MEMORY.md"
    reminder_store_path: str = "state/reminders.json"
    restore_runtime_state_on_startup: bool = False
    reminder_poll_interval_s: float = 1.0
    reminder_retry_delay_s: float = 90.0
    reminder_max_entries: int = 48
    speech_pause_ms: int = 1200
    memory_max_turns: int = 12
    memory_keep_recent: int = 6
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
    display_driver: str = "waveshare_4in2_v2"
    display_vendor_dir: str = "hardware/display/vendor"
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
    display_poll_interval_s: float = 0.5
    printer_queue: str = "Thermal_GP58"
    printer_device_uri: str | None = None
    printer_header_text: str = "TWINR.com"
    printer_feed_lines: int = 3
    printer_line_width: int = 30
    print_button_cooldown_s: float = 2.0
    print_max_lines: int = 8
    print_max_chars: int = 320
    print_context_turns: int = 6

    @property
    def button_gpios(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        if self.green_button_gpio is not None:
            mapping["green"] = self.green_button_gpio
        if self.yellow_button_gpio is not None:
            mapping["yellow"] = self.yellow_button_gpio
        return mapping

    @property
    def pir_enabled(self) -> bool:
        return self.pir_motion_gpio is not None

    @property
    def local_timezone_name(self) -> str:
        return (self.openai_web_search_timezone or "Europe/Berlin").strip() or "Europe/Berlin"

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> "TwinrConfig":
        path = Path(env_path)
        file_values = _read_dotenv(path)
        project_root = path.parent.resolve()

        def get_value(name: str, default: str | None = None) -> str | None:
            if name in os.environ:
                return os.environ[name]
            return file_values.get(name, default)

        return cls(
            openai_api_key=get_value("OPENAI_API_KEY"),
            openai_project_id=get_value("OPENAI_PROJ_ID"),
            openai_send_project_header=_parse_optional_bool(get_value("OPENAI_SEND_PROJECT_HEADER")),
            project_root=str(project_root),
            personality_dir=get_value("TWINR_PERSONALITY_DIR", "personality") or "personality",
            user_display_name=get_value("TWINR_USER_DISPLAY_NAME"),
            default_model=get_value("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
            openai_reasoning_effort=get_value("OPENAI_REASONING_EFFORT", "medium") or "medium",
            openai_stt_model=get_value("OPENAI_STT_MODEL", "whisper-1") or "whisper-1",
            openai_tts_model=get_value("OPENAI_TTS_MODEL", "gpt-4o-mini-tts") or "gpt-4o-mini-tts",
            openai_tts_voice=get_value("OPENAI_TTS_VOICE", "marin") or "marin",
            openai_tts_format=get_value("OPENAI_TTS_FORMAT", "wav") or "wav",
            openai_tts_instructions=get_value("OPENAI_TTS_INSTRUCTIONS"),
            openai_realtime_model=get_value("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
            or "gpt-4o-realtime-preview",
            openai_realtime_voice=get_value("OPENAI_REALTIME_VOICE", "sage") or "sage",
            openai_realtime_instructions=get_value("OPENAI_REALTIME_INSTRUCTIONS"),
            openai_realtime_transcription_model=(
                get_value("OPENAI_REALTIME_TRANSCRIPTION_MODEL", "whisper-1") or "whisper-1"
            ),
            openai_realtime_language=get_value("OPENAI_REALTIME_LANGUAGE", "de") or "de",
            openai_realtime_input_sample_rate=int(
                get_value("OPENAI_REALTIME_INPUT_SAMPLE_RATE", "24000") or "24000"
            ),
            conversation_follow_up_enabled=_parse_bool(
                get_value("TWINR_CONVERSATION_FOLLOW_UP_ENABLED"),
                False,
            ),
            conversation_follow_up_timeout_s=_parse_float(
                get_value("TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S"),
                4.0,
            ),
            audio_beep_frequency_hz=int(get_value("TWINR_AUDIO_BEEP_FREQUENCY_HZ", "1046") or "1046"),
            audio_beep_duration_ms=int(get_value("TWINR_AUDIO_BEEP_DURATION_MS", "180") or "180"),
            audio_beep_volume=_parse_float(get_value("TWINR_AUDIO_BEEP_VOLUME"), 0.8),
            audio_beep_settle_ms=int(get_value("TWINR_AUDIO_BEEP_SETTLE_MS", "120") or "120"),
            search_feedback_tones_enabled=_parse_bool(get_value("TWINR_SEARCH_FEEDBACK_TONES_ENABLED"), True),
            search_feedback_delay_ms=int(get_value("TWINR_SEARCH_FEEDBACK_DELAY_MS", "1200") or "1200"),
            search_feedback_pause_ms=int(get_value("TWINR_SEARCH_FEEDBACK_PAUSE_MS", "900") or "900"),
            search_feedback_volume=_parse_float(get_value("TWINR_SEARCH_FEEDBACK_VOLUME"), 0.14),
            audio_speech_start_chunks=int(get_value("TWINR_AUDIO_SPEECH_START_CHUNKS", "1") or "1"),
            audio_follow_up_speech_start_chunks=int(
                get_value("TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS", "4") or "4"
            ),
            audio_follow_up_ignore_ms=int(get_value("TWINR_AUDIO_FOLLOW_UP_IGNORE_MS", "300") or "300"),
            openai_enable_web_search=_parse_bool(get_value("TWINR_OPENAI_ENABLE_WEB_SEARCH"), False),
            openai_search_model=get_value("OPENAI_SEARCH_MODEL", "gpt-5.2-chat-latest") or "gpt-5.2-chat-latest",
            openai_web_search_context_size=get_value("TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE", "medium") or "medium",
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
            camera_device=get_value("TWINR_CAMERA_DEVICE", "/dev/video0") or "/dev/video0",
            camera_width=int(get_value("TWINR_CAMERA_WIDTH", "640") or "640"),
            camera_height=int(get_value("TWINR_CAMERA_HEIGHT", "480") or "480"),
            camera_framerate=int(get_value("TWINR_CAMERA_FRAMERATE", "30") or "30"),
            camera_input_format=get_value("TWINR_CAMERA_INPUT_FORMAT"),
            camera_ffmpeg_path=get_value("TWINR_CAMERA_FFMPEG_PATH", "ffmpeg") or "ffmpeg",
            vision_reference_image_path=get_value("TWINR_VISION_REFERENCE_IMAGE"),
            openai_vision_detail=get_value("OPENAI_VISION_DETAIL", "auto") or "auto",
            proactive_enabled=_parse_bool(get_value("TWINR_PROACTIVE_ENABLED"), False),
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
            restore_runtime_state_on_startup=_parse_bool(
                get_value("TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP"),
                False,
            ),
            reminder_poll_interval_s=_parse_float(get_value("TWINR_REMINDER_POLL_INTERVAL_S"), 1.0),
            reminder_retry_delay_s=_parse_float(get_value("TWINR_REMINDER_RETRY_DELAY_S"), 90.0),
            reminder_max_entries=int(get_value("TWINR_REMINDER_MAX_ENTRIES", "48") or "48"),
            speech_pause_ms=int(get_value("TWINR_SPEECH_PAUSE_MS", "1200") or "1200"),
            memory_max_turns=int(get_value("TWINR_MEMORY_MAX_TURNS", "12") or "12"),
            memory_keep_recent=int(get_value("TWINR_MEMORY_KEEP_RECENT", "6") or "6"),
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
            display_driver=get_value("TWINR_DISPLAY_DRIVER", "waveshare_4in2_v2") or "waveshare_4in2_v2",
            display_vendor_dir=get_value("TWINR_DISPLAY_VENDOR_DIR", "hardware/display/vendor")
            or "hardware/display/vendor",
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
            display_poll_interval_s=_parse_float(get_value("TWINR_DISPLAY_POLL_INTERVAL_S"), 0.5),
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
