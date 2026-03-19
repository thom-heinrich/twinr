from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig


class TwinrConfigTests(unittest.TestCase):
    def test_parse_float_accepts_optional_bounds(self) -> None:
        from twinr.agent.base_agent.config import _parse_float

        self.assertEqual(_parse_float("0.1", 2.0, minimum=0.25), 0.25)
        self.assertEqual(_parse_float("9.5", 2.0, maximum=5.0), 5.0)
        self.assertEqual(_parse_float(None, 2.0, minimum=1.0), 2.0)
        self.assertEqual(_parse_float(12.5, 2.0, minimum=1.0), 12.5)

    def test_frontier_streaming_defaults_favor_fast_search_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.openai_search_model, "gpt-4o-mini-search-preview")
        self.assertEqual(config.streaming_first_word_model, "gpt-4o-mini")
        self.assertEqual(config.streaming_first_word_context_turns, 1)
        self.assertEqual(config.streaming_first_word_max_output_tokens, 32)
        self.assertEqual(config.streaming_first_word_prefetch_min_chars, 4)
        self.assertEqual(config.streaming_first_word_prefetch_wait_ms, 40)
        self.assertEqual(config.streaming_bridge_reply_timeout_ms, 250)
        self.assertEqual(config.streaming_first_word_final_lane_wait_ms, 900)
        self.assertEqual(config.streaming_final_lane_watchdog_timeout_ms, 4000)
        self.assertEqual(config.streaming_final_lane_hard_timeout_ms, 15000)
        self.assertEqual(config.streaming_supervisor_prefetch_min_chars, 8)
        self.assertEqual(config.streaming_supervisor_prefetch_wait_ms, 80)
        self.assertEqual(config.streaming_specialist_model, "gpt-4o-mini")
        self.assertEqual(config.streaming_specialist_reasoning_effort, "low")

    def test_display_gpio_conflicts_reports_button_overlap(self) -> None:
        config = TwinrConfig(
            display_driver="waveshare_4in2_v2",
            green_button_gpio=23,
            yellow_button_gpio=24,
            display_cs_gpio=8,
            display_dc_gpio=25,
            display_reset_gpio=17,
            display_busy_gpio=24,
        )

        self.assertEqual(
            config.display_gpio_conflicts(),
            ("Display BUSY GPIO 24 collides with yellow button GPIO 24.",),
        )

    def test_hdmi_display_driver_skips_gpio_conflicts(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_fbdev",
            green_button_gpio=23,
            yellow_button_gpio=24,
            display_busy_gpio=24,
        )

        self.assertFalse(config.display_uses_gpio)
        self.assertEqual(config.display_gpio_conflicts(), ())

    def test_from_env_normalizes_display_layout_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("TWINR_DISPLAY_LAYOUT=DEBUG_FACE\n", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.display_layout, "debug_log")

    def test_from_env_reads_display_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("TWINR_DISPLAY_BUSY_TIMEOUT_S=12.5\n", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.display_busy_timeout_s, 12.5)

    def test_from_env_reads_display_runtime_trace_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("TWINR_DISPLAY_RUNTIME_TRACE_ENABLED=true\n", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertTrue(config.display_runtime_trace_enabled)

    def test_from_env_reads_wayland_display_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_DISPLAY_DRIVER=hdmi_wayland",
                        "TWINR_DISPLAY_WAYLAND_DISPLAY=wayland-1",
                        "TWINR_DISPLAY_WAYLAND_RUNTIME_DIR=/run/user/1001",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.display_driver, "hdmi_wayland")
        self.assertEqual(config.display_wayland_display, "wayland-1")
        self.assertEqual(config.display_wayland_runtime_dir, "/run/user/1001")

    def test_from_env_reads_display_companion_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "TWINR_DISPLAY_COMPANION_ENABLED=true\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertTrue(config.display_companion_enabled)

    def test_from_env_reads_display_face_cue_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_DISPLAY_FACE_CUE_PATH=state/custom/face.json",
                        "TWINR_DISPLAY_FACE_CUE_TTL_S=7.5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.display_face_cue_path, "state/custom/face.json")
        self.assertEqual(config.display_face_cue_ttl_s, 7.5)

    def test_from_env_reads_display_presentation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_DISPLAY_PRESENTATION_PATH=state/custom/presentation.json",
                        "TWINR_DISPLAY_PRESENTATION_TTL_S=18.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.display_presentation_path, "state/custom/presentation.json")
        self.assertEqual(config.display_presentation_ttl_s, 18.0)

    def test_from_env_reads_display_news_ticker_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_DISPLAY_NEWS_TICKER_ENABLED=true",
                        "TWINR_DISPLAY_NEWS_TICKER_FEED_URLS=https://example.com/a.rss, https://example.com/b.atom",
                        "TWINR_DISPLAY_NEWS_TICKER_STORE_PATH=state/custom/news.json",
                        "TWINR_DISPLAY_NEWS_TICKER_REFRESH_INTERVAL_S=1200",
                        "TWINR_DISPLAY_NEWS_TICKER_ROTATION_INTERVAL_S=15",
                        "TWINR_DISPLAY_NEWS_TICKER_MAX_ITEMS=8",
                        "TWINR_DISPLAY_NEWS_TICKER_TIMEOUT_S=3.5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertTrue(config.display_news_ticker_enabled)
        self.assertEqual(
            config.display_news_ticker_feed_urls,
            ("https://example.com/a.rss", "https://example.com/b.atom"),
        )
        self.assertEqual(config.display_news_ticker_store_path, "state/custom/news.json")
        self.assertEqual(config.display_news_ticker_refresh_interval_s, 1200.0)
        self.assertEqual(config.display_news_ticker_rotation_interval_s, 15.0)
        self.assertEqual(config.display_news_ticker_max_items, 8)
        self.assertEqual(config.display_news_ticker_timeout_s, 3.5)

    def test_from_env_reads_whatsapp_channel_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_WHATSAPP_ALLOW_FROM=+49 171 1234567",
                        "TWINR_WHATSAPP_NODE_BINARY=/usr/bin/node",
                        f"TWINR_WHATSAPP_AUTH_DIR={root / 'state' / 'channels' / 'whatsapp' / 'auth'}",
                        f"TWINR_WHATSAPP_WORKER_ROOT={root / 'src' / 'twinr' / 'channels' / 'whatsapp' / 'worker'}",
                        "TWINR_WHATSAPP_GROUPS_ENABLED=true",
                        "TWINR_WHATSAPP_SELF_CHAT_MODE=true",
                        "TWINR_WHATSAPP_RECONNECT_BASE_DELAY_S=3.5",
                        "TWINR_WHATSAPP_RECONNECT_MAX_DELAY_S=44",
                        "TWINR_WHATSAPP_SEND_TIMEOUT_S=25",
                        "TWINR_WHATSAPP_SENT_CACHE_TTL_S=240",
                        "TWINR_WHATSAPP_SENT_CACHE_MAX_ENTRIES=512",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.whatsapp_allow_from, "+49 171 1234567")
        self.assertEqual(config.whatsapp_node_binary, "/usr/bin/node")
        self.assertTrue(config.whatsapp_groups_enabled)
        self.assertTrue(config.whatsapp_self_chat_mode)
        self.assertEqual(config.whatsapp_reconnect_base_delay_s, 3.5)
        self.assertEqual(config.whatsapp_reconnect_max_delay_s, 44.0)
        self.assertEqual(config.whatsapp_send_timeout_s, 25.0)
        self.assertEqual(config.whatsapp_sent_cache_ttl_s, 240.0)
        self.assertEqual(config.whatsapp_sent_cache_max_entries, 512)

    def test_from_env_reads_local_camera_mediapipe_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_PROACTIVE_LOCAL_CAMERA_POSE_BACKEND=mediapipe",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_POSE_MODEL_PATH=state/mediapipe/models/pose.task",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_HAND_LANDMARKER_MODEL_PATH=state/mediapipe/models/hand.task",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_GESTURE_MODEL_PATH=state/mediapipe/models/gesture.task",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_CUSTOM_GESTURE_MODEL_PATH=state/mediapipe/models/custom.task",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_MEDIAPIPE_NUM_HANDS=1",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_SEQUENCE_WINDOW_S=2.4",
                        "TWINR_PROACTIVE_LOCAL_CAMERA_SEQUENCE_MIN_FRAMES=6",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.proactive_local_camera_pose_backend, "mediapipe")
        self.assertEqual(
            config.proactive_local_camera_mediapipe_pose_model_path,
            "state/mediapipe/models/pose.task",
        )
        self.assertEqual(
            config.proactive_local_camera_mediapipe_hand_landmarker_model_path,
            "state/mediapipe/models/hand.task",
        )
        self.assertEqual(
            config.proactive_local_camera_mediapipe_gesture_model_path,
            "state/mediapipe/models/gesture.task",
        )
        self.assertEqual(
            config.proactive_local_camera_mediapipe_custom_gesture_model_path,
            "state/mediapipe/models/custom.task",
        )
        self.assertEqual(config.proactive_local_camera_mediapipe_num_hands, 1)
        self.assertEqual(config.proactive_local_camera_sequence_window_s, 2.4)
        self.assertEqual(config.proactive_local_camera_sequence_min_frames, 6)

    def test_reads_openai_button_and_printer_settings_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_REASONING_EFFORT=medium",
                        "OPENAI_SEND_PROJECT_HEADER=false",
                        "OPENAI_PROMPT_CACHE_ENABLED=true",
                        "OPENAI_PROMPT_CACHE_RETENTION=24h",
                        "TWINR_STT_PROVIDER=deepgram",
                        "TWINR_LLM_PROVIDER=groq",
                        "TWINR_TTS_PROVIDER=openai",
                        "TWINR_USER_DISPLAY_NAME=Thom",
                        "OPENAI_STT_MODEL=whisper-1",
                        "OPENAI_TTS_MODEL=gpt-4o-mini-tts",
                        "OPENAI_TTS_VOICE=marin",
                        "OPENAI_TTS_SPEED=0.90",
                        "OPENAI_TTS_FORMAT=wav",
                        "OPENAI_TTS_INSTRUCTIONS=Speak in natural German.",
                        "DEEPGRAM_API_KEY=deepgram-key",
                        "DEEPGRAM_BASE_URL=https://api.deepgram.example/v1",
                        "DEEPGRAM_STT_MODEL=nova-3",
                        "DEEPGRAM_STT_LANGUAGE=de",
                        "DEEPGRAM_STT_SMART_FORMAT=false",
                        "DEEPGRAM_STREAMING_INTERIM_RESULTS=false",
                        "DEEPGRAM_STREAMING_ENDPOINTING_MS=550",
                        "DEEPGRAM_STREAMING_UTTERANCE_END_MS=1200",
                        "DEEPGRAM_STREAMING_STOP_ON_UTTERANCE_END=false",
                        "DEEPGRAM_STREAMING_FINALIZE_TIMEOUT_S=6.5",
                        "DEEPGRAM_TIMEOUT_S=22.0",
                        "GROQ_API_KEY=groq-key",
                        "GROQ_BASE_URL=https://api.groq.example/openai/v1",
                        "GROQ_MODEL=llama-3.3-70b-versatile",
                        "GROQ_TIMEOUT_S=33.0",
                        "OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview",
                        "OPENAI_REALTIME_VOICE=sage",
                        "OPENAI_REALTIME_SPEED=1.05",
                        "OPENAI_REALTIME_INSTRUCTIONS=Speak concise German.",
                        "OPENAI_REALTIME_TRANSCRIPTION_MODEL=whisper-1",
                        "OPENAI_REALTIME_LANGUAGE=de",
                        "OPENAI_REALTIME_INPUT_SAMPLE_RATE=24000",
                        "TWINR_TURN_CONTROLLER_ENABLED=false",
                        "TWINR_TURN_CONTROLLER_CONTEXT_TURNS=6",
                        "TWINR_TURN_CONTROLLER_INSTRUCTIONS_FILE=custom-turn-controller.md",
                        "TWINR_TURN_CONTROLLER_FAST_ENDPOINT_ENABLED=false",
                        "TWINR_TURN_CONTROLLER_FAST_ENDPOINT_MIN_CHARS=14",
                        "TWINR_TURN_CONTROLLER_FAST_ENDPOINT_MIN_CONFIDENCE=0.82",
                        "TWINR_TURN_CONTROLLER_BACKCHANNEL_MAX_CHARS=18",
                        "TWINR_TURN_CONTROLLER_INTERRUPT_ENABLED=false",
                        "TWINR_TURN_CONTROLLER_INTERRUPT_WINDOW_MS=360",
                        "TWINR_TURN_CONTROLLER_INTERRUPT_POLL_MS=75",
                        "TWINR_TURN_CONTROLLER_INTERRUPT_MIN_ACTIVE_RATIO=0.25",
                        "TWINR_TURN_CONTROLLER_INTERRUPT_MIN_TRANSCRIPT_CHARS=6",
                        "TWINR_TURN_CONTROLLER_INTERRUPT_CONSECUTIVE_WINDOWS=3",
                        "TWINR_STREAMING_EARLY_TRANSCRIPT_ENABLED=false",
                        "TWINR_STREAMING_EARLY_TRANSCRIPT_MIN_CHARS=18",
                        "TWINR_STREAMING_EARLY_TRANSCRIPT_WAIT_MS=420",
                        "TWINR_STREAMING_TRANSCRIPT_VERIFIER_ENABLED=false",
                        "TWINR_STREAMING_TRANSCRIPT_VERIFIER_MODEL=gpt-4o-transcribe",
                        "TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_WORDS=4",
                        "TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_CHARS=20",
                        "TWINR_STREAMING_TRANSCRIPT_VERIFIER_MIN_CONFIDENCE=0.88",
                        "TWINR_STREAMING_TRANSCRIPT_VERIFIER_MAX_CAPTURE_MS=4200",
                        "TWINR_CONVERSATION_FOLLOW_UP_ENABLED=true",
                        "TWINR_CONVERSATION_FOLLOW_UP_AFTER_PROACTIVE_ENABLED=true",
                        "TWINR_CONVERSATION_CLOSURE_GUARD_ENABLED=false",
                        "TWINR_CONVERSATION_CLOSURE_CONTEXT_TURNS=3",
                        "TWINR_CONVERSATION_CLOSURE_INSTRUCTIONS_FILE=ALT_CONVERSATION_CLOSURE.md",
                        "TWINR_CONVERSATION_CLOSURE_PROVIDER_TIMEOUT_SECONDS=1.25",
                        "TWINR_CONVERSATION_CLOSURE_MAX_TRANSCRIPT_CHARS=333",
                        "TWINR_CONVERSATION_CLOSURE_MAX_RESPONSE_CHARS=444",
                        "TWINR_CONVERSATION_CLOSURE_MAX_REASON_CHARS=111",
                        "TWINR_CONVERSATION_CLOSURE_MIN_CONFIDENCE=0.72",
                        "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S=3.5",
                        "TWINR_AUDIO_BEEP_FREQUENCY_HZ=1175",
                        "TWINR_AUDIO_BEEP_DURATION_MS=220",
                        "TWINR_AUDIO_BEEP_VOLUME=0.9",
                        "TWINR_AUDIO_BEEP_SETTLE_MS=150",
                        "TWINR_PROCESSING_FEEDBACK_DELAY_MS=25",
                        "TWINR_SEARCH_FEEDBACK_TONES_ENABLED=false",
                        "TWINR_SEARCH_FEEDBACK_DELAY_MS=1100",
                        "TWINR_SEARCH_FEEDBACK_PAUSE_MS=650",
                        "TWINR_SEARCH_FEEDBACK_VOLUME=0.22",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_ENABLED=false",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_UTTERANCE_MAX_MS=900",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_LONG_UTTERANCE_MIN_MS=4200",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_PAUSE_BONUS_MS=150",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_SHORT_PAUSE_GRACE_BONUS_MS=40",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_MEDIUM_PAUSE_PENALTY_MS=90",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_MEDIUM_PAUSE_GRACE_PENALTY_MS=180",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_LONG_PAUSE_PENALTY_MS=280",
                        "TWINR_AUDIO_DYNAMIC_PAUSE_LONG_PAUSE_GRACE_PENALTY_MS=180",
                        "TWINR_AUDIO_PAUSE_RESUME_CHUNKS=3",
                        "TWINR_AUDIO_SPEECH_START_CHUNKS=3",
                        "TWINR_AUDIO_FOLLOW_UP_SPEECH_START_CHUNKS=5",
                        "TWINR_AUDIO_FOLLOW_UP_IGNORE_MS=420",
                        "TWINR_OPENAI_ENABLE_WEB_SEARCH=true",
                        "OPENAI_SEARCH_MODEL=gpt-5.2",
                        "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE=high",
                        "TWINR_OPENAI_SEARCH_MAX_OUTPUT_TOKENS=180",
                        "TWINR_OPENAI_SEARCH_RETRY_MAX_OUTPUT_TOKENS=260",
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
                        "TWINR_STREAMING_TTS_CLAUSE_MIN_CHARS=32",
                        "TWINR_STREAMING_TTS_SOFT_SEGMENT_CHARS=84",
                        "TWINR_STREAMING_TTS_HARD_SEGMENT_CHARS=132",
                        "OPENAI_TTS_STREAM_CHUNK_SIZE=1024",
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
                        "TWINR_PROACTIVE_VISION_REVIEW_ENABLED=true",
                        "TWINR_PROACTIVE_VISION_REVIEW_BUFFER_FRAMES=9",
                        "TWINR_PROACTIVE_VISION_REVIEW_MAX_FRAMES=5",
                        "TWINR_PROACTIVE_VISION_REVIEW_MAX_AGE_S=15.5",
                        "TWINR_PROACTIVE_VISION_REVIEW_MIN_SPACING_S=1.7",
                        "TWINR_WAKEWORD_ENABLED=true",
                        "TWINR_WAKEWORD_BACKEND=openwakeword",
                        "TWINR_WAKEWORD_PRIMARY_BACKEND=openwakeword",
                        "TWINR_WAKEWORD_FALLBACK_BACKEND=stt",
                        "TWINR_WAKEWORD_VERIFIER_MODE=ambiguity_only",
                        "TWINR_WAKEWORD_VERIFIER_MARGIN=0.11",
                        "TWINR_WAKEWORD_PHRASES=hey twinr, hey twinna, twinr, twinner",
                        "TWINR_WAKEWORD_SAMPLE_MS=1700",
                        "TWINR_WAKEWORD_PRESENCE_GRACE_S=600",
                        "TWINR_WAKEWORD_MOTION_GRACE_S=180",
                        "TWINR_WAKEWORD_SPEECH_GRACE_S=75",
                        "TWINR_WAKEWORD_ATTEMPT_COOLDOWN_S=5.5",
                        "TWINR_WAKEWORD_BLOCK_PROACTIVE_AFTER_ATTEMPT_S=14",
                        "TWINR_WAKEWORD_MIN_ACTIVE_RATIO=0.06",
                        "TWINR_WAKEWORD_MIN_ACTIVE_CHUNKS=2",
                        "TWINR_WAKEWORD_OPENWAKEWORD_MODELS=/twinr/models/wakewords/twinr.tflite, /twinr/models/wakewords/twinna.tflite",
                        "TWINR_WAKEWORD_OPENWAKEWORD_THRESHOLD=0.57",
                        "TWINR_WAKEWORD_OPENWAKEWORD_VAD_THRESHOLD=0.18",
                        "TWINR_WAKEWORD_OPENWAKEWORD_PATIENCE_FRAMES=3",
                        "TWINR_WAKEWORD_OPENWAKEWORD_ACTIVATION_SAMPLES=4",
                        "TWINR_WAKEWORD_OPENWAKEWORD_DEACTIVATION_THRESHOLD=0.12",
                        "TWINR_WAKEWORD_OPENWAKEWORD_ENABLE_SPEEX=true",
                        "TWINR_WAKEWORD_OPENWAKEWORD_TRANSCRIBE_ON_DETECT=false",
                        "TWINR_WAKEWORD_OPENWAKEWORD_INFERENCE_FRAMEWORK=onnx",
                        "TWINR_WAKEWORD_CALIBRATION_PROFILE_PATH=state/custom_wakeword_calibration.json",
                        "TWINR_WAKEWORD_CALIBRATION_RECOMMENDED_PATH=state/custom_wakeword_calibration.recommended.json",
                        "TWINR_PROACTIVE_PERSON_RETURNED_ABSENCE_S=1400",
                        "TWINR_PROACTIVE_PERSON_RETURNED_RECENT_MOTION_S=45",
                        "TWINR_PROACTIVE_ATTENTION_WINDOW_S=7.5",
                        "TWINR_PROACTIVE_SLUMPED_QUIET_S=24",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_STILLNESS_S=12",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_VISIBILITY_LOSS_HOLD_S=18",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_VISIBILITY_LOSS_ARMING_S=6.5",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_SLUMPED_VISIBILITY_LOSS_ARMING_S=4.5",
                        "TWINR_PROACTIVE_POSSIBLE_FALL_ONCE_PER_PRESENCE_SESSION=false",
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
                        "TWINR_PROACTIVE_GOVERNOR_ENABLED=false",
                        "TWINR_PROACTIVE_GOVERNOR_ACTIVE_RESERVATION_TTL_S=33",
                        "TWINR_PROACTIVE_GOVERNOR_GLOBAL_PROMPT_COOLDOWN_S=180",
                        "TWINR_PROACTIVE_GOVERNOR_WINDOW_S=2400",
                        "TWINR_PROACTIVE_GOVERNOR_WINDOW_PROMPT_LIMIT=5",
                        "TWINR_PROACTIVE_GOVERNOR_PRESENCE_SESSION_PROMPT_LIMIT=3",
                        "TWINR_PROACTIVE_GOVERNOR_SOURCE_REPEAT_COOLDOWN_S=420",
                        "TWINR_PROACTIVE_GOVERNOR_HISTORY_LIMIT=96",
                        "TWINR_PROACTIVE_VISUAL_FIRST_AUDIO_GLOBAL_COOLDOWN_S=240",
                        "TWINR_PROACTIVE_VISUAL_FIRST_AUDIO_SOURCE_REPEAT_COOLDOWN_S=840",
                        "TWINR_PROACTIVE_VISUAL_FIRST_CUE_HOLD_S=75",
                        "TWINR_PROACTIVE_QUIET_HOURS_VISUAL_ONLY_ENABLED=false",
                        "TWINR_PROACTIVE_QUIET_HOURS_START_LOCAL=22:15",
                        "TWINR_PROACTIVE_QUIET_HOURS_END_LOCAL=06:45",
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
                        "TWINR_LONG_TERM_MEMORY_MODE=remote_primary",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_REQUIRED=true",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_NAMESPACE=pi-main",
                        "TWINR_LONG_TERM_MEMORY_PATH=/tmp/twinr-chonkydb",
                        "TWINR_LONG_TERM_MEMORY_BACKGROUND_STORE_TURNS=false",
                        "TWINR_LONG_TERM_MEMORY_WRITE_QUEUE_SIZE=48",
                        "TWINR_LONG_TERM_MEMORY_RECALL_LIMIT=5",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_READ_TIMEOUT_S=5.5",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_WRITE_TIMEOUT_S=11.5",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_KEEPALIVE_INTERVAL_S=2.25",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_INTERVAL_S=1.5",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_WATCHDOG_HISTORY_LIMIT=7200",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_RETRY_ATTEMPTS=4",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_RETRY_BACKOFF_S=2.5",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_FLUSH_TIMEOUT_S=75",
                        "TWINR_LONG_TERM_MEMORY_TURN_EXTRACTOR_MODEL=gpt-5.2-mini",
                        "TWINR_LONG_TERM_MEMORY_TURN_EXTRACTOR_MAX_OUTPUT_TOKENS=2600",
                        "TWINR_LONG_TERM_MEMORY_MIDTERM_ENABLED=false",
                        "TWINR_LONG_TERM_MEMORY_MIDTERM_LIMIT=6",
                        "TWINR_LONG_TERM_MEMORY_REFLECTION_WINDOW_SIZE=24",
                        "TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_ENABLED=false",
                        "TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_MODEL=gpt-5.2-nano",
                        "TWINR_LONG_TERM_MEMORY_REFLECTION_COMPILER_MAX_OUTPUT_TOKENS=640",
                        "TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_ENABLED=false",
                        "TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_MODEL=gpt-5.2-mini",
                        "TWINR_LONG_TERM_MEMORY_SUBTEXT_COMPILER_MAX_OUTPUT_TOKENS=196",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_POLL_INTERVAL_S=18.0",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_MIN_CONFIDENCE=0.81",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_REPEAT_COOLDOWN_S=28800",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_SKIP_COOLDOWN_S=900",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_RESERVATION_TTL_S=75",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_ALLOW_SENSITIVE=true",
                        "TWINR_LONG_TERM_MEMORY_PROACTIVE_HISTORY_LIMIT=64",
                        "TWINR_LONG_TERM_MEMORY_SENSOR_MEMORY_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_SENSOR_BASELINE_DAYS=28",
                        "TWINR_LONG_TERM_MEMORY_SENSOR_MIN_DAYS_OBSERVED=7",
                        "TWINR_LONG_TERM_MEMORY_SENSOR_MIN_ROUTINE_RATIO=0.66",
                        "TWINR_LONG_TERM_MEMORY_SENSOR_DEVIATION_MIN_DELTA=0.52",
                        "TWINR_LONG_TERM_MEMORY_RETENTION_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_RETENTION_MODE=conservative",
                        "TWINR_LONG_TERM_MEMORY_RETENTION_RUN_INTERVAL_S=420",
                        "TWINR_LONG_TERM_MEMORY_ARCHIVE_ENABLED=true",
                        "TWINR_LONG_TERM_MEMORY_MIGRATION_ENABLED=false",
                        "TWINR_LONG_TERM_MEMORY_MIGRATION_BATCH_SIZE=32",
                        "TWINR_LONG_TERM_MEMORY_REMOTE_BULK_REQUEST_MAX_BYTES=131072",
                        "TWINR_CHONKYDB_BASE_URL=https://memory.example.com:2149",
                        "TWINR_CHONKYDB_API_KEY=secret-key",
                        "TWINR_CHONKYDB_API_KEY_HEADER=x-api-key",
                        "TWINR_CHONKYDB_ALLOW_BEARER_AUTH=true",
                        "TWINR_CHONKYDB_TIMEOUT_S=14.5",
                        "TWINR_CHONKYDB_MAX_RESPONSE_BYTES=25165824",
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
                        "TWINR_DISPLAY_DRIVER=hdmi_fbdev",
                        "TWINR_DISPLAY_FB_PATH=/dev/fb1",
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
        self.assertTrue(config.openai_prompt_cache_enabled)
        self.assertEqual(config.openai_prompt_cache_retention, "24h")
        self.assertEqual(config.stt_provider, "deepgram")
        self.assertEqual(config.llm_provider, "groq")
        self.assertEqual(config.tts_provider, "openai")
        self.assertEqual(config.user_display_name, "Thom")
        self.assertEqual(config.openai_stt_model, "whisper-1")
        self.assertEqual(config.openai_tts_model, "gpt-4o-mini-tts")
        self.assertEqual(config.openai_tts_voice, "marin")
        self.assertEqual(config.openai_tts_speed, 0.9)
        self.assertEqual(config.openai_tts_format, "wav")
        self.assertEqual(config.openai_tts_instructions, "Speak in natural German.")
        self.assertEqual(config.deepgram_api_key, "deepgram-key")
        self.assertEqual(config.deepgram_base_url, "https://api.deepgram.example/v1")
        self.assertEqual(config.deepgram_stt_model, "nova-3")
        self.assertEqual(config.deepgram_stt_language, "de")
        self.assertFalse(config.deepgram_stt_smart_format)
        self.assertFalse(config.deepgram_streaming_interim_results)
        self.assertEqual(config.deepgram_streaming_endpointing_ms, 550)
        self.assertEqual(config.deepgram_streaming_utterance_end_ms, 1200)
        self.assertFalse(config.deepgram_streaming_stop_on_utterance_end)
        self.assertEqual(config.deepgram_streaming_finalize_timeout_s, 6.5)
        self.assertEqual(config.deepgram_timeout_s, 22.0)
        self.assertEqual(config.groq_api_key, "groq-key")
        self.assertEqual(config.groq_base_url, "https://api.groq.example/openai/v1")
        self.assertEqual(config.groq_model, "llama-3.3-70b-versatile")
        self.assertEqual(config.groq_timeout_s, 33.0)
        self.assertEqual(config.openai_realtime_model, "gpt-4o-realtime-preview")
        self.assertEqual(config.openai_realtime_voice, "sage")
        self.assertEqual(config.openai_realtime_speed, 1.05)
        self.assertEqual(config.openai_realtime_instructions, "Speak concise German.")
        self.assertEqual(config.openai_realtime_transcription_model, "whisper-1")
        self.assertEqual(config.openai_realtime_language, "de")
        self.assertEqual(config.openai_realtime_input_sample_rate, 24000)
        self.assertFalse(config.turn_controller_enabled)
        self.assertEqual(config.turn_controller_context_turns, 6)
        self.assertEqual(config.turn_controller_instructions_file, "custom-turn-controller.md")
        self.assertFalse(config.turn_controller_fast_endpoint_enabled)
        self.assertEqual(config.turn_controller_fast_endpoint_min_chars, 14)
        self.assertEqual(config.turn_controller_fast_endpoint_min_confidence, 0.82)
        self.assertEqual(config.turn_controller_backchannel_max_chars, 18)
        self.assertFalse(config.turn_controller_interrupt_enabled)
        self.assertEqual(config.turn_controller_interrupt_window_ms, 360)
        self.assertEqual(config.turn_controller_interrupt_poll_ms, 75)
        self.assertEqual(config.turn_controller_interrupt_min_active_ratio, 0.25)
        self.assertEqual(config.turn_controller_interrupt_min_transcript_chars, 6)
        self.assertEqual(config.turn_controller_interrupt_consecutive_windows, 3)
        self.assertFalse(config.streaming_early_transcript_enabled)
        self.assertEqual(config.streaming_early_transcript_min_chars, 18)
        self.assertEqual(config.streaming_early_transcript_wait_ms, 420)
        self.assertFalse(config.streaming_transcript_verifier_enabled)
        self.assertEqual(config.streaming_transcript_verifier_model, "gpt-4o-transcribe")
        self.assertEqual(config.streaming_transcript_verifier_max_words, 4)
        self.assertEqual(config.streaming_transcript_verifier_max_chars, 20)
        self.assertEqual(config.streaming_transcript_verifier_min_confidence, 0.88)
        self.assertEqual(config.streaming_transcript_verifier_max_capture_ms, 4200)
        self.assertTrue(config.conversation_follow_up_enabled)
        self.assertTrue(config.conversation_follow_up_after_proactive_enabled)
        self.assertFalse(config.conversation_closure_guard_enabled)
        self.assertEqual(config.conversation_closure_context_turns, 3)
        self.assertEqual(config.conversation_closure_instructions_file, "ALT_CONVERSATION_CLOSURE.md")
        self.assertEqual(config.conversation_closure_provider_timeout_seconds, 1.25)
        self.assertEqual(config.conversation_closure_max_transcript_chars, 333)
        self.assertEqual(config.conversation_closure_max_response_chars, 444)
        self.assertEqual(config.conversation_closure_max_reason_chars, 111)
        self.assertEqual(config.conversation_closure_min_confidence, 0.72)
        self.assertEqual(config.conversation_follow_up_timeout_s, 3.5)
        self.assertEqual(config.audio_beep_frequency_hz, 1175)
        self.assertEqual(config.audio_beep_duration_ms, 220)
        self.assertEqual(config.audio_beep_volume, 0.9)
        self.assertEqual(config.audio_beep_settle_ms, 150)
        self.assertEqual(config.processing_feedback_delay_ms, 25)
        self.assertFalse(config.search_feedback_tones_enabled)
        self.assertEqual(config.search_feedback_delay_ms, 1100)
        self.assertEqual(config.search_feedback_pause_ms, 650)
        self.assertEqual(config.search_feedback_volume, 0.22)
        self.assertFalse(config.audio_dynamic_pause_enabled)
        self.assertEqual(config.audio_dynamic_pause_short_utterance_max_ms, 900)
        self.assertEqual(config.audio_dynamic_pause_long_utterance_min_ms, 4200)
        self.assertEqual(config.audio_dynamic_pause_short_pause_bonus_ms, 150)
        self.assertEqual(config.audio_dynamic_pause_short_pause_grace_bonus_ms, 40)
        self.assertEqual(config.audio_dynamic_pause_medium_pause_penalty_ms, 90)
        self.assertEqual(config.audio_dynamic_pause_medium_pause_grace_penalty_ms, 180)
        self.assertEqual(config.audio_dynamic_pause_long_pause_penalty_ms, 280)
        self.assertEqual(config.audio_dynamic_pause_long_pause_grace_penalty_ms, 180)
        self.assertEqual(config.audio_pause_resume_chunks, 3)
        self.assertEqual(config.audio_speech_start_chunks, 3)
        self.assertEqual(config.audio_follow_up_speech_start_chunks, 5)
        self.assertEqual(config.audio_follow_up_ignore_ms, 420)
        self.assertTrue(config.openai_enable_web_search)
        self.assertEqual(config.openai_search_model, "gpt-5.2")
        self.assertEqual(config.openai_web_search_context_size, "high")
        self.assertEqual(config.openai_search_max_output_tokens, 180)
        self.assertEqual(config.openai_search_retry_max_output_tokens, 260)
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
        self.assertEqual(config.streaming_tts_clause_min_chars, 32)
        self.assertEqual(config.streaming_tts_soft_segment_chars, 84)
        self.assertEqual(config.streaming_tts_hard_segment_chars, 132)
        self.assertEqual(config.openai_tts_stream_chunk_size, 1024)
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
        self.assertTrue(config.proactive_vision_review_enabled)
        self.assertEqual(config.proactive_vision_review_buffer_frames, 9)
        self.assertEqual(config.proactive_vision_review_max_frames, 5)
        self.assertEqual(config.proactive_vision_review_max_age_s, 15.5)
        self.assertEqual(config.proactive_vision_review_min_spacing_s, 1.7)
        self.assertTrue(config.wakeword_enabled)
        self.assertEqual(config.wakeword_backend, "openwakeword")
        self.assertEqual(config.wakeword_primary_backend, "openwakeword")
        self.assertEqual(config.wakeword_fallback_backend, "stt")
        self.assertEqual(config.wakeword_verifier_mode, "ambiguity_only")
        self.assertEqual(config.wakeword_verifier_margin, 0.11)
        self.assertEqual(config.wakeword_phrases, ("hey twinr", "hey twinna", "twinr", "twinner"))
        self.assertEqual(config.wakeword_sample_ms, 1700)
        self.assertEqual(config.wakeword_presence_grace_s, 600.0)
        self.assertEqual(config.wakeword_motion_grace_s, 180.0)
        self.assertEqual(config.wakeword_speech_grace_s, 75.0)
        self.assertEqual(config.wakeword_attempt_cooldown_s, 5.5)
        self.assertEqual(config.wakeword_block_proactive_after_attempt_s, 14.0)
        self.assertEqual(config.wakeword_min_active_ratio, 0.06)
        self.assertEqual(config.wakeword_min_active_chunks, 2)
        self.assertEqual(
            config.wakeword_openwakeword_models,
            ("/twinr/models/wakewords/twinr.tflite", "/twinr/models/wakewords/twinna.tflite"),
        )
        self.assertEqual(config.wakeword_openwakeword_threshold, 0.57)
        self.assertEqual(config.wakeword_openwakeword_vad_threshold, 0.18)
        self.assertEqual(config.wakeword_openwakeword_patience_frames, 3)
        self.assertEqual(config.wakeword_openwakeword_activation_samples, 4)
        self.assertEqual(config.wakeword_openwakeword_deactivation_threshold, 0.12)
        self.assertTrue(config.wakeword_openwakeword_enable_speex)
        self.assertFalse(config.wakeword_openwakeword_transcribe_on_detect)
        self.assertEqual(config.wakeword_openwakeword_inference_framework, "onnx")
        self.assertEqual(
            config.wakeword_calibration_profile_path,
            "state/custom_wakeword_calibration.json",
        )
        self.assertEqual(
            config.wakeword_calibration_recommended_path,
            "state/custom_wakeword_calibration.recommended.json",
        )
        self.assertEqual(config.proactive_person_returned_absence_s, 1400.0)
        self.assertEqual(config.proactive_person_returned_recent_motion_s, 45.0)
        self.assertEqual(config.proactive_attention_window_s, 7.5)
        self.assertEqual(config.proactive_slumped_quiet_s, 24.0)
        self.assertEqual(config.proactive_possible_fall_stillness_s, 12.0)
        self.assertEqual(config.proactive_possible_fall_visibility_loss_hold_s, 18.0)
        self.assertEqual(config.proactive_possible_fall_visibility_loss_arming_s, 6.5)
        self.assertEqual(config.proactive_possible_fall_slumped_visibility_loss_arming_s, 4.5)
        self.assertFalse(config.proactive_possible_fall_once_per_presence_session)
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
        self.assertFalse(config.proactive_governor_enabled)
        self.assertEqual(config.proactive_governor_active_reservation_ttl_s, 33.0)
        self.assertEqual(config.proactive_governor_global_prompt_cooldown_s, 180.0)
        self.assertEqual(config.proactive_governor_window_s, 2400.0)
        self.assertEqual(config.proactive_governor_window_prompt_limit, 5)
        self.assertEqual(config.proactive_governor_presence_session_prompt_limit, 3)
        self.assertEqual(config.proactive_governor_source_repeat_cooldown_s, 420.0)
        self.assertEqual(config.proactive_governor_history_limit, 96)
        self.assertEqual(config.proactive_visual_first_audio_global_cooldown_s, 240.0)
        self.assertEqual(config.proactive_visual_first_audio_source_repeat_cooldown_s, 840.0)
        self.assertEqual(config.proactive_visual_first_cue_hold_s, 75.0)
        self.assertFalse(config.proactive_quiet_hours_visual_only_enabled)
        self.assertEqual(config.proactive_quiet_hours_start_local, "22:15")
        self.assertEqual(config.proactive_quiet_hours_end_local, "06:45")
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
        self.assertEqual(config.long_term_memory_mode, "remote_primary")
        self.assertTrue(config.long_term_memory_remote_required)
        self.assertEqual(config.long_term_memory_remote_namespace, "pi-main")
        self.assertEqual(config.long_term_memory_path, "/tmp/twinr-chonkydb")
        self.assertFalse(config.long_term_memory_background_store_turns)
        self.assertEqual(config.long_term_memory_write_queue_size, 48)
        self.assertEqual(config.long_term_memory_recall_limit, 5)
        self.assertEqual(config.long_term_memory_remote_read_timeout_s, 5.5)
        self.assertEqual(config.long_term_memory_remote_write_timeout_s, 11.5)
        self.assertEqual(config.long_term_memory_remote_keepalive_interval_s, 2.25)
        self.assertEqual(config.long_term_memory_remote_watchdog_interval_s, 1.5)
        self.assertEqual(config.long_term_memory_remote_watchdog_history_limit, 7200)
        self.assertEqual(config.long_term_memory_remote_retry_attempts, 4)
        self.assertEqual(config.long_term_memory_remote_retry_backoff_s, 2.5)
        self.assertEqual(config.long_term_memory_remote_flush_timeout_s, 75.0)
        self.assertEqual(config.long_term_memory_turn_extractor_model, "gpt-5.2-mini")
        self.assertEqual(config.long_term_memory_turn_extractor_max_output_tokens, 2600)
        self.assertFalse(config.long_term_memory_midterm_enabled)
        self.assertEqual(config.long_term_memory_midterm_limit, 6)
        self.assertEqual(config.long_term_memory_reflection_window_size, 24)
        self.assertFalse(config.long_term_memory_reflection_compiler_enabled)
        self.assertEqual(config.long_term_memory_reflection_compiler_model, "gpt-5.2-nano")
        self.assertEqual(config.long_term_memory_reflection_compiler_max_output_tokens, 640)
        self.assertFalse(config.long_term_memory_subtext_compiler_enabled)
        self.assertEqual(config.long_term_memory_subtext_compiler_model, "gpt-5.2-mini")
        self.assertEqual(config.long_term_memory_subtext_compiler_max_output_tokens, 196)
        self.assertTrue(config.long_term_memory_proactive_enabled)
        self.assertEqual(config.long_term_memory_proactive_poll_interval_s, 18.0)
        self.assertEqual(config.long_term_memory_proactive_min_confidence, 0.81)
        self.assertEqual(config.long_term_memory_proactive_repeat_cooldown_s, 28800.0)
        self.assertEqual(config.long_term_memory_proactive_skip_cooldown_s, 900.0)
        self.assertEqual(config.long_term_memory_proactive_reservation_ttl_s, 75.0)
        self.assertTrue(config.long_term_memory_proactive_allow_sensitive)
        self.assertEqual(config.long_term_memory_proactive_history_limit, 64)
        self.assertTrue(config.long_term_memory_sensor_memory_enabled)
        self.assertEqual(config.long_term_memory_sensor_baseline_days, 28)
        self.assertEqual(config.long_term_memory_sensor_min_days_observed, 7)
        self.assertEqual(config.long_term_memory_sensor_min_routine_ratio, 0.66)
        self.assertEqual(config.long_term_memory_sensor_deviation_min_delta, 0.52)
        self.assertTrue(config.long_term_memory_retention_enabled)
        self.assertEqual(config.long_term_memory_retention_mode, "conservative")
        self.assertEqual(config.long_term_memory_retention_run_interval_s, 420.0)
        self.assertTrue(config.long_term_memory_archive_enabled)
        self.assertFalse(config.long_term_memory_migration_enabled)
        self.assertEqual(config.long_term_memory_migration_batch_size, 32)
        self.assertEqual(config.long_term_memory_remote_bulk_request_max_bytes, 131072)
        self.assertEqual(config.chonkydb_base_url, "https://memory.example.com:2149")
        self.assertEqual(config.chonkydb_api_key, "secret-key")
        self.assertEqual(config.chonkydb_api_key_header, "x-api-key")
        self.assertTrue(config.chonkydb_allow_bearer_auth)
        self.assertEqual(config.chonkydb_timeout_s, 14.5)
        self.assertEqual(config.chonkydb_max_response_bytes, 25165824)
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
        self.assertEqual(config.display_driver, "hdmi_fbdev")
        self.assertEqual(config.display_fb_path, "/dev/fb1")
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
        self.assertEqual(config.display_busy_timeout_s, 20.0)
        self.assertFalse(config.display_runtime_trace_enabled)
        self.assertEqual(config.printer_queue, "Twinr_Test_Printer")
        self.assertEqual(
            config.printer_device_uri,
            "usb://Gprinter/GP-58?serial=WTTING%20",
        )
        self.assertEqual(config.printer_header_text, "TWINR.com")
        self.assertEqual(config.printer_feed_lines, 5)
        self.assertEqual(config.printer_line_width, 28)
        self.assertEqual(config.print_button_cooldown_s, 2.5)

    def test_display_vendor_dir_defaults_to_state_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.display_vendor_dir, "state/display/vendor")

    def test_remote_primary_always_forces_fail_closed_remote_semantics(self) -> None:
        config = TwinrConfig(
            long_term_memory_enabled=True,
            long_term_memory_mode="remote_primary",
            long_term_memory_remote_required=False,
        )

        self.assertEqual(config.long_term_memory_mode, "remote_primary")
        self.assertTrue(config.long_term_memory_remote_required)

    def test_defaults_raise_memory_capacity(self) -> None:
        config = TwinrConfig()

        self.assertEqual(config.memory_max_turns, 20)
        self.assertEqual(config.memory_keep_recent, 10)
        self.assertTrue(config.adaptive_timing_enabled)
        self.assertEqual(config.adaptive_timing_pause_grace_ms, 450)
        self.assertEqual(config.processing_feedback_delay_ms, 0)
        self.assertTrue(config.turn_controller_enabled)
        self.assertEqual(config.turn_controller_context_turns, 4)
        self.assertEqual(config.display_full_refresh_interval, 0)
        self.assertEqual(config.turn_controller_instructions_file, "TURN_CONTROLLER.md")
        self.assertTrue(config.turn_controller_fast_endpoint_enabled)
        self.assertEqual(config.turn_controller_fast_endpoint_min_chars, 10)
        self.assertEqual(config.turn_controller_fast_endpoint_min_confidence, 0.9)
        self.assertEqual(config.turn_controller_backchannel_max_chars, 24)
        self.assertTrue(config.turn_controller_interrupt_enabled)
        self.assertEqual(config.turn_controller_interrupt_window_ms, 420)
        self.assertEqual(config.turn_controller_interrupt_poll_ms, 120)
        self.assertEqual(config.turn_controller_interrupt_min_active_ratio, 0.18)
        self.assertEqual(config.turn_controller_interrupt_min_transcript_chars, 4)
        self.assertEqual(config.turn_controller_interrupt_consecutive_windows, 2)
        self.assertTrue(config.streaming_early_transcript_enabled)
        self.assertEqual(config.streaming_early_transcript_min_chars, 10)
        self.assertEqual(config.streaming_early_transcript_wait_ms, 250)
        self.assertTrue(config.streaming_transcript_verifier_enabled)
        self.assertEqual(config.streaming_transcript_verifier_model, "gpt-4o-mini-transcribe")
        self.assertEqual(config.streaming_transcript_verifier_max_words, 6)
        self.assertEqual(config.streaming_transcript_verifier_max_chars, 32)
        self.assertEqual(config.streaming_transcript_verifier_min_confidence, 0.92)
        self.assertFalse(config.display_runtime_trace_enabled)
        self.assertEqual(config.streaming_transcript_verifier_max_capture_ms, 6500)
        self.assertEqual(config.streaming_first_word_prefetch_min_words, 2)
        self.assertTrue(config.conversation_closure_guard_enabled)
        self.assertEqual(config.conversation_closure_context_turns, 4)
        self.assertEqual(config.conversation_closure_instructions_file, "CONVERSATION_CLOSURE.md")
        self.assertEqual(config.conversation_closure_provider_timeout_seconds, 2.0)
        self.assertEqual(config.conversation_closure_max_transcript_chars, 512)
        self.assertEqual(config.conversation_closure_max_response_chars, 512)
        self.assertEqual(config.conversation_closure_max_reason_chars, 256)
        self.assertEqual(config.conversation_closure_min_confidence, 0.65)
        self.assertTrue(config.audio_dynamic_pause_enabled)
        self.assertEqual(config.audio_dynamic_pause_short_utterance_max_ms, 1000)
        self.assertEqual(config.audio_dynamic_pause_long_utterance_min_ms, 5000)
        self.assertEqual(config.audio_dynamic_pause_short_pause_bonus_ms, 120)
        self.assertEqual(config.audio_dynamic_pause_short_pause_grace_bonus_ms, 0)
        self.assertEqual(config.audio_dynamic_pause_medium_pause_penalty_ms, 120)
        self.assertEqual(config.audio_dynamic_pause_medium_pause_grace_penalty_ms, 250)
        self.assertEqual(config.audio_dynamic_pause_long_pause_penalty_ms, 320)
        self.assertEqual(config.audio_dynamic_pause_long_pause_grace_penalty_ms, 220)
        self.assertEqual(config.audio_pause_resume_chunks, 2)
        self.assertFalse(config.long_term_memory_enabled)
        self.assertEqual(config.long_term_memory_backend, "chonkydb")
        self.assertEqual(config.long_term_memory_mode, "local_first")
        self.assertFalse(config.long_term_memory_remote_required)
        self.assertIsNone(config.long_term_memory_remote_namespace)
        self.assertEqual(config.long_term_memory_path, "state/chonkydb")
        self.assertTrue(config.long_term_memory_background_store_turns)
        self.assertEqual(config.long_term_memory_write_queue_size, 32)
        self.assertEqual(config.long_term_memory_recall_limit, 3)
        self.assertEqual(config.long_term_memory_remote_read_timeout_s, 8.0)
        self.assertEqual(config.long_term_memory_remote_write_timeout_s, 15.0)
        self.assertEqual(config.long_term_memory_remote_keepalive_interval_s, 5.0)
        self.assertEqual(config.long_term_memory_remote_watchdog_interval_s, 1.0)
        self.assertEqual(config.long_term_memory_remote_watchdog_history_limit, 3600)
        self.assertEqual(config.long_term_memory_remote_retry_attempts, 3)
        self.assertEqual(config.long_term_memory_remote_retry_backoff_s, 1.0)
        self.assertEqual(config.long_term_memory_remote_flush_timeout_s, 60.0)
        self.assertIsNone(config.long_term_memory_turn_extractor_model)
        self.assertEqual(config.long_term_memory_turn_extractor_max_output_tokens, 2200)
        self.assertTrue(config.long_term_memory_midterm_enabled)
        self.assertEqual(config.long_term_memory_midterm_limit, 4)
        self.assertEqual(config.long_term_memory_reflection_window_size, 18)
        self.assertTrue(config.long_term_memory_reflection_compiler_enabled)
        self.assertIsNone(config.long_term_memory_reflection_compiler_model)
        self.assertEqual(config.long_term_memory_reflection_compiler_max_output_tokens, 900)
        self.assertTrue(config.long_term_memory_subtext_compiler_enabled)
        self.assertIsNone(config.long_term_memory_subtext_compiler_model)
        self.assertEqual(config.long_term_memory_subtext_compiler_max_output_tokens, 520)
        self.assertFalse(config.long_term_memory_proactive_enabled)
        self.assertEqual(config.long_term_memory_proactive_poll_interval_s, 30.0)
        self.assertEqual(config.long_term_memory_proactive_min_confidence, 0.72)
        self.assertEqual(config.long_term_memory_proactive_repeat_cooldown_s, 21600.0)
        self.assertEqual(config.long_term_memory_proactive_skip_cooldown_s, 1800.0)
        self.assertEqual(config.long_term_memory_proactive_reservation_ttl_s, 90.0)
        self.assertFalse(config.long_term_memory_proactive_allow_sensitive)
        self.assertEqual(config.long_term_memory_proactive_history_limit, 128)
        self.assertFalse(config.long_term_memory_sensor_memory_enabled)
        self.assertEqual(config.long_term_memory_sensor_baseline_days, 21)
        self.assertEqual(config.long_term_memory_sensor_min_days_observed, 6)
        self.assertEqual(config.long_term_memory_sensor_min_routine_ratio, 0.55)
        self.assertEqual(config.long_term_memory_sensor_deviation_min_delta, 0.45)
        self.assertTrue(config.long_term_memory_retention_enabled)
        self.assertEqual(config.long_term_memory_retention_mode, "conservative")
        self.assertEqual(config.long_term_memory_retention_run_interval_s, 300.0)
        self.assertTrue(config.long_term_memory_archive_enabled)
        self.assertTrue(config.long_term_memory_migration_enabled)
        self.assertEqual(config.long_term_memory_migration_batch_size, 64)
        self.assertEqual(config.long_term_memory_remote_bulk_request_max_bytes, 512 * 1024)
        self.assertEqual(config.proactive_possible_fall_visibility_loss_hold_s, 15.0)
        self.assertEqual(config.proactive_possible_fall_visibility_loss_arming_s, 6.0)
        self.assertEqual(config.proactive_possible_fall_slumped_visibility_loss_arming_s, 4.0)
        self.assertTrue(config.proactive_possible_fall_once_per_presence_session)
        self.assertTrue(config.proactive_governor_enabled)
        self.assertEqual(config.proactive_governor_active_reservation_ttl_s, 45.0)
        self.assertEqual(config.proactive_governor_global_prompt_cooldown_s, 120.0)
        self.assertEqual(config.proactive_governor_window_s, 1200.0)
        self.assertEqual(config.proactive_governor_window_prompt_limit, 4)
        self.assertEqual(config.proactive_governor_presence_session_prompt_limit, 2)
        self.assertEqual(config.proactive_governor_source_repeat_cooldown_s, 600.0)
        self.assertEqual(config.proactive_governor_history_limit, 128)
        self.assertEqual(config.proactive_visual_first_audio_global_cooldown_s, 300.0)
        self.assertEqual(config.proactive_visual_first_audio_source_repeat_cooldown_s, 900.0)
        self.assertEqual(config.proactive_visual_first_cue_hold_s, 45.0)
        self.assertTrue(config.proactive_quiet_hours_visual_only_enabled)
        self.assertEqual(config.proactive_quiet_hours_start_local, "21:00")
        self.assertEqual(config.proactive_quiet_hours_end_local, "07:00")

    def test_wakeword_openwakeword_threshold_allows_deployment_tuned_low_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_WAKEWORD_ENABLED=true",
                        "TWINR_WAKEWORD_BACKEND=openwakeword",
                        "TWINR_WAKEWORD_OPENWAKEWORD_THRESHOLD=0.001",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.wakeword_openwakeword_threshold, 0.001)

    def test_wakeword_transcribe_on_detect_legacy_flag_maps_to_verifier_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_WAKEWORD_ENABLED=true",
                        "TWINR_WAKEWORD_BACKEND=openwakeword",
                        "TWINR_WAKEWORD_OPENWAKEWORD_TRANSCRIBE_ON_DETECT=true",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.wakeword_primary_backend, "openwakeword")
        self.assertEqual(config.wakeword_verifier_mode, "always")

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

    def test_chonkydb_response_limit_defaults_to_64_mib(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.chonkydb_max_response_bytes, 64 * 1024 * 1024)

    def test_streaming_dual_lane_env_settings_are_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "TWINR_STREAMING_DUAL_LANE_ENABLED=false",
                        "TWINR_STREAMING_FIRST_WORD_ENABLED=false",
                        "TWINR_STREAMING_FIRST_WORD_MODEL=gpt-4o-mini",
                        "TWINR_STREAMING_FIRST_WORD_REASONING_EFFORT=low",
                        "TWINR_STREAMING_FIRST_WORD_CONTEXT_TURNS=1",
                        "TWINR_STREAMING_FIRST_WORD_MAX_OUTPUT_TOKENS=36",
                        "TWINR_STREAMING_FIRST_WORD_PREFETCH_ENABLED=false",
                        "TWINR_STREAMING_FIRST_WORD_PREFETCH_MIN_CHARS=5",
                        "TWINR_STREAMING_FIRST_WORD_PREFETCH_MIN_WORDS=3",
                        "TWINR_STREAMING_FIRST_WORD_PREFETCH_WAIT_MS=45",
                        "TWINR_STREAMING_BRIDGE_REPLY_TIMEOUT_MS=300",
                        "TWINR_STREAMING_FINAL_LANE_WATCHDOG_TIMEOUT_MS=4500",
                        "TWINR_STREAMING_FINAL_LANE_HARD_TIMEOUT_MS=16000",
                        "TWINR_STREAMING_SUPERVISOR_MODEL=gpt-4o-mini",
                        "TWINR_STREAMING_SUPERVISOR_REASONING_EFFORT=low",
                        "TWINR_STREAMING_SPECIALIST_MODEL=gpt-5.2-chat-latest",
                        "TWINR_STREAMING_SPECIALIST_REASONING_EFFORT=medium",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertFalse(config.streaming_dual_lane_enabled)
        self.assertFalse(config.streaming_first_word_enabled)
        self.assertEqual(config.streaming_first_word_model, "gpt-4o-mini")
        self.assertEqual(config.streaming_first_word_reasoning_effort, "low")
        self.assertEqual(config.streaming_first_word_context_turns, 1)
        self.assertEqual(config.streaming_first_word_max_output_tokens, 36)
        self.assertFalse(config.streaming_first_word_prefetch_enabled)
        self.assertEqual(config.streaming_first_word_prefetch_min_chars, 5)
        self.assertEqual(config.streaming_first_word_prefetch_min_words, 3)
        self.assertEqual(config.streaming_first_word_prefetch_wait_ms, 45)
        self.assertEqual(config.streaming_bridge_reply_timeout_ms, 300)
        self.assertEqual(config.streaming_final_lane_watchdog_timeout_ms, 4500)
        self.assertEqual(config.streaming_final_lane_hard_timeout_ms, 16000)
        self.assertEqual(config.streaming_supervisor_model, "gpt-4o-mini")
        self.assertEqual(config.streaming_supervisor_reasoning_effort, "low")
        self.assertEqual(config.streaming_specialist_model, "gpt-5.2-chat-latest")
        self.assertEqual(config.streaming_specialist_reasoning_effort, "medium")

    def test_reads_orchestrator_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENAI_API_KEY=test-key",
                        "TWINR_ORCHESTRATOR_HOST=127.0.0.1",
                        "TWINR_ORCHESTRATOR_PORT=9876",
                        "TWINR_ORCHESTRATOR_WS_URL=ws://10.0.0.5:9876/ws/orchestrator",
                        "TWINR_ORCHESTRATOR_SHARED_SECRET=secret-token",
                    ]
                ),
                encoding="utf-8",
            )

            config = TwinrConfig.from_env(env_path)

        self.assertEqual(config.orchestrator_host, "127.0.0.1")
        self.assertEqual(config.orchestrator_port, 9876)
        self.assertEqual(config.orchestrator_ws_url, "ws://10.0.0.5:9876/ws/orchestrator")
        self.assertEqual(config.orchestrator_shared_secret, "secret-token")
