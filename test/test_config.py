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
                        "TWINR_AUDIO_SPEECH_START_CHUNKS=3",
                        "TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS=5",
                        "TWINR_AUDIO_FOLLOW_UP_IGNORE_MS=420",
                        "TWINR_OPENAI_ENABLE_WEB_SEARCH=true",
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
                        "TWINR_GREEN_BUTTON_GPIO=17",
                        "TWINR_YELLOW_BUTTON_GPIO=27",
                        "TWINR_BUTTON_ACTIVE_LOW=true",
                        "TWINR_BUTTON_BIAS=pull-up",
                        "TWINR_BUTTON_DEBOUNCE_MS=120",
                        "TWINR_BUTTON_PROBE_LINES=17,27,22",
                        "TWINR_PRINTER_QUEUE=Twinr_Test_Printer",
                        "TWINR_PRINTER_DEVICE_URI=usb://Gprinter/GP-58?serial=WTTING%20",
                        "TWINR_PRINTER_FEED_LINES=5",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.openai_reasoning_effort, "medium")
        self.assertFalse(config.openai_send_project_header)
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
        self.assertEqual(config.audio_speech_start_chunks, 3)
        self.assertEqual(config.audio_follow_up_speech_start_chunks, 5)
        self.assertEqual(config.audio_follow_up_ignore_ms, 420)
        self.assertTrue(config.openai_enable_web_search)
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
        self.assertEqual(config.green_button_gpio, 17)
        self.assertEqual(config.yellow_button_gpio, 27)
        self.assertTrue(config.button_active_low)
        self.assertEqual(config.button_bias, "pull-up")
        self.assertEqual(config.button_debounce_ms, 120)
        self.assertEqual(config.button_probe_lines, (17, 27, 22))
        self.assertEqual(config.printer_queue, "Twinr_Test_Printer")
        self.assertEqual(
            config.printer_device_uri,
            "usb://Gprinter/GP-58?serial=WTTING%20",
        )
        self.assertEqual(config.printer_feed_lines, 5)
