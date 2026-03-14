from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig


class TwinrConfigTests(unittest.TestCase):
    def test_reads_openai_button_and_printer_settings_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_REASONING_EFFORT=medium",
                        "OPENAI_SEND_PROJECT_HEADER=false",
                        "TWINR_USER_DISPLAY_NAME=Thom",
                        "OPENAI_STT_MODEL=whisper-1",
                        "OPENAI_TTS_MODEL=gpt-4o-mini-tts",
                        "OPENAI_TTS_VOICE=marin",
                        "OPENAI_TTS_SPEED=0.90",
                        "OPENAI_TTS_FORMAT=wav",
                        "OPENAI_TTS_INSTRUCTIONS=Speak in natural German.",
                        "OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview",
                        "OPENAI_REALTIME_VOICE=sage",
                        "OPENAI_REALTIME_SPEED=1.05",
                        "OPENAI_REALTIME_INSTRUCTIONS=Speak concise German.",
                        "OPENAI_REALTIME_TRANSCRIPTION_MODEL=whisper-1",
                        "OPENAI_REALTIME_LANGUAGE=de",
                        "OPENAI_REALTIME_INPUT_SAMPLE_RATE=24000",
                        "TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true",
                        "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S=3.5",
                        "TWINR_AUDIO_BEEP_FREQUENCY_HZ=1175",
                        "TWINR_AUDIO_BEEP_DURATION_MS=220",
                        "TWINR_AUDIO_BEEP_VOLUME=0.9",
                        "TWINR_AUDIO_BEEP_SETTLE_MS=150",
                        "TWINR_SEARCH_FEEDBACK_TONES_ENABLED=false",
                        "TWINR_SEARCH_FEEDBACK_DELAY_MS=1100",
                        "TWINR_SEARCH_FEEDBACK_PAUSE_MS=650",
                        "TWINR_SEARCH_FEEDBACK_VOLUME=0.22",
                        "TWINR_AUDIO_SPEECH_START_CHUNKS=3",
                        "TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS=5",
                        "TWINR_AUDIO_FOLLOW_UP_IGNORE_MS=420",
                        "TWINR_OPENAI_ENABLE_WEB_SEARCH=true",
                        "OPENAI_SEARCH_MODEL=gpt-5.2",
                        "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE=high",
                        "TWINR_OPENAI_WEB_SEARCH_COUNTRY=DE",
                        "TWINR_OPENAI_WEB_SEARCH_REGION=Berlin",
                        "TWINR_OPENAI_WEB_SEARCH_CITY=Berlin",
                        "TWINR_OPENAI_WEB_SEARCH_TIMEZONE=Europe/Berlin",
                        "TWINR_CONVERSATION_WEB_SEARCH=always",
                        "TWINR_AUDIO_INPUT_DEVICE=default",
                        "TWINR_AUDIO_OUTPUT_DEVICE=default",
                        "TWINR_AUDIO_SAMPLE_RATE=22050",
                        "TWINR_AUDIO_CHANNELS=1",
                        "TWINR_AUDIO_CHUNK_MS=80",
                        "TWINR_AUDIO_PREROLL_MS=240",
                        "TWINR_AUDIO_SPEECH_THRESHOLD=950",
                        "TWINR_AUDIO_START_TIMEOUT_S=6.5",
                        "TWINR_AUDIO_MAX_RECORD_SECONDS=18.0",
                        "TWINR_CAMERA_DEVICE=/dev/video2",
                        "TWINR_CAMERA_WIDTH=800",
                        "TWINR_CAMERA_HEIGHT=600",
                        "TWINR_CAMERA_FRAMERATE=25",
                        "TWINR_CAMERA_INPUT_FORMAT=bayer_grbg8",
                        "TWINR_CAMERA_FFMPEG_PATH=/usr/local/bin/ffmpeg",
                        "TWINR_VISION_REFERENCE_IMAGE=/srv/twinr/user-reference.jpg",
                        "OPENAI_VISION_DETAIL=high",
                        "TWINR_PROACTIVE_ENABLED=true",
                        "TWINR_PROACTIVE_POLL_INTERVAL_S=3.0",
                        "TWINR_PROACTIVE_CAPTURE_INTERVAL_S=5.5",
                        "TWINR_PROACTIVE_MOTION_WINDOW_S=18.0",
                        "TWINR_PROACTIVE_LOW_MOTION_AFTER_S=11.5",
                        "TWINR_PROACTIVE_AUDIO_ENABLED=true",
                        "TWINR_PROACTIVE_AUDIO_DEVICE=plughw:CARD=CameraB409241,DEV=0",
                        "TWINR_PROACTIVE_AUDIO_SAMPLE_MS=900",
                        "TWINR_PROACTIVE_AUDIO_DISTRESS_ENABLED=true",
                        "TWINR_WAKEWORD_ENABLED=true",
                        "TWINR_WAKEWORD_BACKEND=openwakeword",
                        "TWINR_WAKEWORD_PHRASES=hey twinr, hey twinna, twinr, twinner",
                        "TWINR_WAKEWORD_SAMPLE_MS=1700",
                        "TWINR_WAKEWORD_PRESENCE_GRACE_S=600",
                        "TWINR_WAKEWORD_MOTION_GRACE_S=180",
                        "TWINR_WAKEWORD_SPEECH_GRACE_S=75",
                        "TWINR_WAKEWORD_ATTEMPT_COOLDOWN_S=5.5",
                        "TWINR_WAKEWORD_MIN_ACTIVE_RATIO=0.06",
                        "TWINR_WAKEWORD_MIN_ACTIVE_CHUNKS=2",
                        "TWINR_WAKEWORD_OPENWAKEWORD_MODELS=/twinr/models/wakewords/twinr.tflite, /twinr/models/wakewords/twinna.tflite",
                        "TWINR_WAKEWORD_OPENWAKEWORD_THRESHOLD=0.57",
                        "TWINR_WAKEWORD_OPENWAKEWORD_VAD_THRESHOLD=0.18",
                        "TWINR_WAKEWORD_OPENWAKEWORD_PATIENCE_FRAMES=3",
                        "TWINR_WAKEWORD_OPENWAKEWORD_ENABLE_SPEEX=true",
                        "TWINR_WAKEWORD_OPENWAKEWORD_TRANSCRIBE_ON_DETECT=false",
                        "TWINR_WAKEWORD_OPENWAKEWORD_INFERENCE_FRAMEWORK=onnx",
                        "TWINR_PROACTIVE_PERSON_RETURNED_ABSENCE_S=1400",
                        "TWINR_PROACTIVE_PERSON_RETURNED_RECENT_MOTION_S=45",
                        "TWINR_PROACTIVE_ATTENTION_WINDOW_S=7.5",
                        "TWINR_PROACTIVE_SLUMPED_QUIET_S=24",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_STILLNESS_S=12",
                        "TWINR_PROACTIVE_FLOOR_STILLNESS_S=28",
                        "TWINR_PROACTIVE_SHOWING_INTENT_HOLD_S=2.2",
                        "TWINR_PROACTIVE_POSITIVE_CONTACT_HOLD_S=2.4",
                        "TWINR_PROACTIVE_DISTRESS_HOLD_S=4.5",
                        "TWINR_PROACTIVE_FALL_TRANSITION_WINDOW_S=9.5",
                        "TWINR_PROACTIVE_PERSON_RETURNED_SCORE_THRESHOLD=0.92",
                        "TWINR_PROACTIVE_ATTENTION_WINDOW_SCORE_THRESHOLD=0.8",
                        "TWINR_PROACTIVE_SLUMPED_QUIET_SCORE_THRESHOLD=0.88",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_SCORE_THRESHOLD=0.77",
                        "TWINR_PROACTIVE_FLOOR_STILLNESS_SCORE_THRESHOLD=0.94",
                        "TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD=0.81",
                        "TWINR_PROACTIVE_POSITIVE_CONTACT_SCORE_THRESHOLD=0.79",
                        "TWINR_PROACTIVE_DISTRESS_POSSIBLE_SCORE_THRESHOLD=0.83",
                        "TWINR_WEB_HOST=0.0.0.0",
                        "TWINR_RUNTIME_STATE_PATH=/tmp/twinr-state-test.json",
                        "TWINR_MEMORY_MARKDOWN_PATH=/tmp/twinr-memory-test.md",
                        "TWINR_REMINDER_STORE_PATH=/tmp/twinr-reminders-test.json",
                        "TWINR_AUTOMATION_STORE_PATH=/tmp/twinr-automations-test.json",
                        "TWINR_VOICE_PROFILE_STORE_PATH=/tmp/twinr-voice-profile.json",
                        "TWINR_ADAPTIVE_TIMING_ENABLED=false",
                        "TWINR_ADAPTIVE_TIMING_STORE_PATH=/tmp/twinr-adaptive-timing.json",
                        "TWINR_ADAPTIVE_TIMING_PAUSE_GRACE_MS=1050",
                        "TWINR_LONG_TERM_MEMORY_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_BACKEND=chonkydb",
                        "TWINR_LONG_TERM_MEMORY_PATH=/tmp/twinr-chonkydb",
                        "TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS=false",
                        "TWINR_LONG_TERM_MEMORY_WRITE_QUEUE_SIZE=48",
                        "TWINR_LONG_TERM_MEMORY_RECALL_LIMIT=5",
                        "TWINR_CHONKYDB_BASE_URL=https://memory.example.com:2149",
                        "TWINR_CHONKYDB_API_KEY=secret-key",
                        "TWINR_CHONKYDB_API_KEY_HEADER=x-api-key",
                        "TWINR_CHONKYDB_ALLOW_BEARER_AUTH=true",
                        "TWINR_CHONKYDB_TIMEOUT_S=14.5",
                        "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP=true",
                        "TWINR_REMINDER_POLL_INTERVAL_S=2.5",
                        "TWINR_REMINDER_RETRY_DELAY_S=45.0",
                        "TWINR_REMINDER_MAX_ENTRIES=72",
                        "TWINR_AUTOMATION_POLL_INTERVAL_S=7.5",
                        "TWINR_AUTOMATION_MAX_ENTRIES=144",
                        "TWINR_VOICE_PROFILE_MIN_SAMPLE_MS=1500",
                        "TWINR_VOICE_PROFILE_LIKELY_THRESHOLD=0.78",
                        "TWINR_VOICE_PROFILE_UNCERTAIN_THRESHOLD=0.60",
                        "TWINR_VOICE_PROFILE_MAX_SAMPLES=8",
                        "TWINR_MEMORY_MAX_TURNS=24",
                        "TWINR_MEMORY_KEEP_RECENT=12",
                        "TWINR_GREEN_BUTTON_GPIO=17",
                        "TWINR_YELLOW_BUTTON_GPIO=27",
                        "TWINR_PIR_MOTION_GPIO=26",
                        "TWINR_PIR_ACTIVE_HIGH=true",
                        "TWINR_PIR_BIAS=pull-down",
                        "TWINR_PIR_DEBOUNCE_MS=150",
                        "TWINR_BUTTON_ACTIVE_LOW=true",
                        "TWINR_BUTTON_BIAS=pull-up",
                        "TWINR_BUTTON_DEBOUNCE_MS=120",
                        "TWINR_BUTTON_PROBE_LINES=17,27,22",
                        "TWINR_DISPLAY_DRIVER=waveshare_4in2_v2",
                        "TWINR_DISPLAY_VENDOR_DIR=hardware/display/vendor",
                        "TWINR_DISPLAY_SPI_BUS=0",
                        "TWINR_DISPLAY_SPI_DEVICE=0",
                        "TWINR_DISPLAY_CS_GPIO=8",
                        "TWINR_DISPLAY_DC_GPIO=25",
                        "TWINR_DISPLAY_RESET_GPIO=17",
                        "TWINR_DISPLAY_BUSY_GPIO=24",
                        "TWINR_DISPLAY_WIDTH=400",
                        "TWINR_DISPLAY_HEIGHT=300",
                        "TWINR_DISPLAY_ROTATION_DEGREES=270",
                        "TWINR_DISPLAY_FULL_REFRESH_INTERVAL=120",
                        "TWINR_PRINTER_QUEUE=Twinr_Test_Printer",
                        "TWINR_PRINTER_DEVICE_URI=usb://Gprinter/GP-58?serial=WTTING%20",
                        "TWINR_PRINTER_HEADER_TEXT=TWINR.com",
                        "TWINR_PRINTER_FEED_LINES=5",
                        "TWINR_PRINTER_LINE_WIDTH=28",
                        "TWINR_PRINT_BUTTON_COOLDOWN_S=2.5",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.openai_reasoning_effort, "medium")
        self.assertFalse(config.openai_send_project_header)
        self.assertEqual(config.user_display_name, "Thom")
        self.assertEqual(config.openai_stt_model, "whisper-1")
        self.assertEqual(config.openai_tts_model, "gpt-4o-mini-tts")
        self.assertEqual(config.openai_tts_voice, "marin")
        self.assertEqual(config.openai_tts_speed, 0.9)
        self.assertEqual(config.openai_tts_format, "wav")
        self.assertEqual(config.openai_tts_instructions, "Speak in natural German.")
        self.assertEqual(config.openai_realtime_model, "gpt-4o-realtime-preview")
        self.assertEqual(config.openai_realtime_voice, "sage")
        self.assertEqual(config.openai_realtime_speed, 1.05)
        self.assertEqual(config.openai_realtime_instructions, "Speak concise German.")
        self.assertEqual(config.openai_realtime_transcription_model, "whisper-1")
        self.assertEqual(config.openai_realtime_language, "de")
        self.assertEqual(config.openai_realtime_input_sample_rate, 24000)
        self.assertTrue(config.conversation_follow_up_enabled)
        self.assertEqual(config.conversation_follow_up_timeout_s, 3.5)
        self.assertEqual(config.audio_beep_frequency_hz, 1175)
        self.assertEqual(config.audio_beep_duration_ms, 220)
        self.assertEqual(config.audio_beep_volume, 0.9)
        self.assertEqual(config.audio_beep_settle_ms, 150)
        self.assertFalse(config.search_feedback_tones_enabled)
        self.assertEqual(config.search_feedback_delay_ms, 1100)
        self.assertEqual(config.search_feedback_pause_ms, 650)
        self.assertEqual(config.search_feedback_volume, 0.22)
        self.assertEqual(config.audio_speech_start_chunks, 3)
        self.assertEqual(config.audio_follow_up_speech_start_chunks, 5)
        self.assertEqual(config.audio_follow_up_ignore_ms, 420)
        self.assertTrue(config.openai_enable_web_search)
        self.assertEqual(config.openai_search_model, "gpt-5.2")
        self.assertEqual(config.openai_web_search_context_size, "high")
        self.assertEqual(config.openai_web_search_country, "DE")
        self.assertEqual(config.openai_web_search_region, "Berlin")
        self.assertEqual(config.openai_web_search_city, "Berlin")
        self.assertEqual(config.openai_web_search_timezone, "Europe/Berlin")
        self.assertEqual(config.conversation_web_search, "always")
        self.assertEqual(config.audio_input_device, "default")
        self.assertEqual(config.audio_output_device, "default")
        self.assertEqual(config.audio_sample_rate, 22050)
        self.assertEqual(config.audio_channels, 1)
        self.assertEqual(config.audio_chunk_ms, 80)
        self.assertEqual(config.audio_preroll_ms, 240)
        self.assertEqual(config.audio_speech_threshold, 950)
        self.assertEqual(config.audio_start_timeout_s, 6.5)
        self.assertEqual(config.audio_max_record_seconds, 18.0)
        self.assertEqual(config.camera_device, "/dev/video2")
        self.assertEqual(config.camera_width, 800)
        self.assertEqual(config.camera_height, 600)
        self.assertEqual(config.camera_framerate, 25)
        self.assertEqual(config.camera_input_format, "bayer_grbg8")
        self.assertEqual(config.camera_ffmpeg_path, "/usr/local/bin/ffmpeg")
        self.assertEqual(config.vision_reference_image_path, "/srv/twinr/user-reference.jpg")
        self.assertEqual(config.openai_vision_detail, "high")
        self.assertTrue(config.proactive_enabled)
        self.assertEqual(config.proactive_poll_interval_s, 3.0)
        self.assertEqual(config.proactive_capture_interval_s, 5.5)
        self.assertEqual(config.proactive_motion_window_s, 18.0)
        self.assertEqual(config.proactive_low_motion_after_s, 11.5)
        self.assertTrue(config.proactive_audio_enabled)
        self.assertEqual(config.proactive_audio_input_device, "plughw:CARD=CameraB409241,DEV=0")
        self.assertEqual(config.proactive_audio_sample_ms, 900)
        self.assertTrue(config.proactive_audio_distress_enabled)
        self.assertTrue(config.wakeword_enabled)
        self.assertEqual(config.wakeword_backend, "openwakeword")
        self.assertEqual(config.wakeword_phrases, ("hey twinr", "hey twinna", "twinr", "twinner"))
        self.assertEqual(config.wakeword_sample_ms, 1700)
        self.assertEqual(config.wakeword_presence_grace_s, 600.0)
        self.assertEqual(config.wakeword_motion_grace_s, 180.0)
        self.assertEqual(config.wakeword_speech_grace_s, 75.0)
        self.assertEqual(config.wakeword_attempt_cooldown_s, 5.5)
        self.assertEqual(config.wakeword_min_active_ratio, 0.06)
        self.assertEqual(config.wakeword_min_active_chunks, 2)
        self.assertEqual(
            config.wakeword_openwakeword_models,
            ("/twinr/models/wakewords/twinr.tflite", "/twinr/models/wakewords/twinna.tflite"),
        )
        self.assertEqual(config.wakeword_openwakeword_threshold, 0.57)
        self.assertEqual(config.wakeword_openwakeword_vad_threshold, 0.18)
        self.assertEqual(config.wakeword_openwakeword_patience_frames, 3)
        self.assertTrue(config.wakeword_openwakeword_enable_speex)
        self.assertFalse(config.wakeword_openwakeword_transcribe_on_detect)
        self.assertEqual(config.wakeword_openwakeword_inference_framework, "onnx")
        self.assertEqual(config.proactive_person_returned_absence_s, 1400.0)
        self.assertEqual(config.proactive_person_returned_recent_motion_s, 45.0)
        self.assertEqual(config.proactive_attention_window_s, 7.5)
        self.assertEqual(config.proactive_slumped_quiet_s, 24.0)
        self.assertEqual(config.proactive_possible_fall_stillness_s, 12.0)
        self.assertEqual(config.proactive_floor_stillness_s, 28.0)
        self.assertEqual(config.proactive_showing_intent_hold_s, 2.2)
        self.assertEqual(config.proactive_positive_contact_hold_s, 2.4)
        self.assertEqual(config.proactive_distress_hold_s, 4.5)
        self.assertEqual(config.proactive_fall_transition_window_s, 9.5)
        self.assertEqual(config.proactive_person_returned_score_threshold, 0.92)
        self.assertEqual(config.proactive_attention_window_score_threshold, 0.8)
        self.assertEqual(config.proactive_slumped_quiet_score_threshold, 0.88)
        self.assertEqual(config.proactive_possible_fall_score_threshold, 0.77)
        self.assertEqual(config.proactive_floor_stillness_score_threshold, 0.94)
        self.assertEqual(config.proactive_showing_intent_score_threshold, 0.81)
        self.assertEqual(config.proactive_positive_contact_score_threshold, 0.79)
        self.assertEqual(config.proactive_distress_possible_score_threshold, 0.83)
        self.assertEqual(config.web_host, "0.0.0.0")
        self.assertEqual(config.runtime_state_path, "/tmp/twinr-state-test.json")
        self.assertEqual(config.memory_markdown_path, "/tmp/twinr-memory-test.md")
        self.assertEqual(config.reminder_store_path, "/tmp/twinr-reminders-test.json")
        self.assertEqual(config.automation_store_path, "/tmp/twinr-automations-test.json")
        self.assertEqual(config.voice_profile_store_path, "/tmp/twinr-voice-profile.json")
        self.assertFalse(config.adaptive_timing_enabled)
        self.assertEqual(config.adaptive_timing_store_path, "/tmp/twinr-adaptive-timing.json")
        self.assertEqual(config.adaptive_timing_pause_grace_ms, 1050)
        self.assertTrue(config.long_term_memory_enabled)
        self.assertEqual(config.long_term_memory_backend, "chonkydb")
        self.assertEqual(config.long_term_memory_path, "/tmp/twinr-chonkydb")
        self.assertFalse(config.long_term_memory_background_store_turns)
        self.assertEqual(config.long_term_memory_write_queue_size, 48)
        self.assertEqual(config.long_term_memory_recall_limit, 5)
        self.assertEqual(config.chonkydb_base_url, "https://memory.example.com:2149")
        self.assertEqual(config.chonkydb_api_key, "secret-key")
        self.assertEqual(config.chonkydb_api_key_header, "x-api-key")
        self.assertTrue(config.chonkydb_allow_bearer_auth)
        self.assertEqual(config.chonkydb_timeout_s, 14.5)
        self.assertTrue(config.restore_runtime_state_on_startup)
        self.assertEqual(config.reminder_poll_interval_s, 2.5)
        self.assertEqual(config.reminder_retry_delay_s, 45.0)
        self.assertEqual(config.reminder_max_entries, 72)
        self.assertEqual(config.automation_poll_interval_s, 7.5)
        self.assertEqual(config.automation_max_entries, 144)
        self.assertEqual(config.voice_profile_min_sample_ms, 1500)
        self.assertEqual(config.voice_profile_likely_threshold, 0.78)
        self.assertEqual(config.voice_profile_uncertain_threshold, 0.60)
        self.assertEqual(config.voice_profile_max_samples, 8)
        self.assertEqual(config.memory_max_turns, 24)
        self.assertEqual(config.memory_keep_recent, 12)
        self.assertEqual(config.green_button_gpio, 17)
        self.assertEqual(config.yellow_button_gpio, 27)
        self.assertEqual(config.pir_motion_gpio, 26)
        self.assertTrue(config.pir_active_high)
        self.assertEqual(config.pir_bias, "pull-down")
        self.assertEqual(config.pir_debounce_ms, 150)
        self.assertTrue(config.button_active_low)
        self.assertEqual(config.button_bias, "pull-up")
        self.assertEqual(config.button_debounce_ms, 120)
        self.assertEqual(config.button_probe_lines, (17, 27, 22))
        self.assertEqual(config.display_driver, "waveshare_4in2_v2")
        self.assertEqual(config.display_vendor_dir, "hardware/display/vendor")
        self.assertEqual(config.display_spi_bus, 0)
        self.assertEqual(config.display_spi_device, 0)
        self.assertEqual(config.display_cs_gpio, 8)
        self.assertEqual(config.display_dc_gpio, 25)
        self.assertEqual(config.display_reset_gpio, 17)
        self.assertEqual(config.display_busy_gpio, 24)
        self.assertEqual(config.display_width, 400)
        self.assertEqual(config.display_height, 300)
        self.assertEqual(config.display_rotation_degrees, 270)
        self.assertEqual(config.display_full_refresh_interval, 120)
        self.assertEqual(config.printer_queue, "Twinr_Test_Printer")
        self.assertEqual(
            config.printer_device_uri,
            "usb://Gprinter/GP-58?serial=WTTING%20",
        )
        self.assertEqual(config.printer_header_text, "TWINR.com")
        self.assertEqual(config.printer_feed_lines, 5)
        self.assertEqual(config.printer_line_width, 28)
        self.assertEqual(config.print_button_cooldown_s, 2.5)

    def test_defaults_raise_memory_capacity(self) -> None:
        config = TwinrConfig()

        self.assertEqual(config.memory_max_turns, 20)
        self.assertEqual(config.memory_keep_recent, 10)
        self.assertTrue(config.adaptive_timing_enabled)
        self.assertEqual(config.adaptive_timing_pause_grace_ms, 900)
        self.assertFalse(config.long_term_memory_enabled)
        self.assertEqual(config.long_term_memory_backend, "chonkydb")
        self.assertEqual(config.long_term_memory_path, "state/chonkydb")
        self.assertTrue(config.long_term_memory_background_store_turns)
        self.assertEqual(config.long_term_memory_write_queue_size, 32)
        self.assertEqual(config.long_term_memory_recall_limit, 3)

    def test_from_env_defaults_long_term_memory_path_to_project_local_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.long_term_memory_path, str(Path(temp_dir) / "state" / "chonkydb"))

    def test_legacy_ccodex_envs_feed_chonkydb_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "CCODEX_MEMORY_BASE_URL=https://legacy-memory.example.com:2149",
                        "CCODEX_MEMORY_API_KEY=legacy-secret",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.chonkydb_base_url, "https://legacy-memory.example.com:2149")
        self.assertEqual(config.chonkydb_api_key, "legacy-secret")
