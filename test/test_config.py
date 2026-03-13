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
                        "OPENAI_TTS_FORMAT=wav",
                        "OPENAI_TTS_INSTRUCTIONS=Speak in natural German.",
                        "OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview",
                        "OPENAI_REALTIME_VOICE=sage",
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
                        "TWINR_WEB_HOST=0.0.0.0",
                        "TWINR_RUNTIME_STATE_PATH=/tmp/twinr-state-test.json",
                        "TWINR_MEMORY_MARKDOWN_PATH=/tmp/twinr-memory-test.md",
                        "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP=true",
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
        self.assertEqual(config.openai_tts_format, "wav")
        self.assertEqual(config.openai_tts_instructions, "Speak in natural German.")
        self.assertEqual(config.openai_realtime_model, "gpt-4o-realtime-preview")
        self.assertEqual(config.openai_realtime_voice, "sage")
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
        self.assertEqual(config.web_host, "0.0.0.0")
        self.assertEqual(config.runtime_state_path, "/tmp/twinr-state-test.json")
        self.assertEqual(config.memory_markdown_path, "/tmp/twinr-memory-test.md")
        self.assertTrue(config.restore_runtime_state_on_startup)
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
