from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from twinr.agent.base_agent import RuntimeSnapshotStore, TwinrConfig
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry
from twinr.ops import TwinrOpsEventStore, resolve_ops_paths
from twinr.web import create_app


class WebAppTests(unittest.TestCase):
    def make_client(self) -> tuple[TestClient, Path]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        env_path = root / ".env"
        personality_dir = root / "personality"
        personality_dir.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            "\n".join(
                [
                    "OPENAI_MODEL=gpt-5.2",
                    "OPENAI_API_KEY=sk-test-1234",
                    "TWINR_WEB_HOST=0.0.0.0",
                    "TWINR_WEB_PORT=1337",
                    f"TWINR_RUNTIME_STATE_PATH={root / 'runtime-state.json'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (personality_dir / "SYSTEM.md").write_text("System text\n", encoding="utf-8")
        (personality_dir / "PERSONALITY.md").write_text("Personality text\n", encoding="utf-8")
        (personality_dir / "USER.md").write_text("User text\n", encoding="utf-8")
        return TestClient(create_app(env_path)), env_path

    def test_dashboard_renders_summary(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Twinr", response.text)
        self.assertIn("Dashboard", response.text)
        self.assertIn("sk-t…1234", response.text)
        self.assertIn("Status and failures", response.text)

    def test_connect_page_renders_inline_help(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/connect")

        self.assertEqual(response.status_code, 200)
        self.assertIn("field-tooltip", response.text)
        self.assertIn("The main OpenAI secret used for chat, speech, vision, and realtime requests.", response.text)
        self.assertIn("Controls which backend answers normal text questions.", response.text)
        self.assertIn("(?)", response.text)

    def test_settings_post_updates_env(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/settings",
            data={
                "OPENAI_MODEL": "gpt-4o-mini",
                "OPENAI_STT_MODEL": "whisper-1",
                "OPENAI_TTS_MODEL": "gpt-4o-mini-tts",
                "OPENAI_TTS_VOICE": "marin",
                "OPENAI_REALTIME_MODEL": "gpt-4o-realtime-preview",
                "OPENAI_REALTIME_VOICE": "sage",
                "TWINR_WEB_HOST": "127.0.0.1",
                "TWINR_WEB_PORT": "1440",
                "TWINR_SPEECH_PAUSE_MS": "900",
                "TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S": "3.0",
                "TWINR_CONVERSATION_FOLLOW_UP_ENABLED": "true",
                "TWINR_AUDIO_SPEECH_THRESHOLD": "800",
                "TWINR_AUDIO_BEEP_FREQUENCY_HZ": "1200",
                "TWINR_AUDIO_BEEP_DURATION_MS": "180",
                "TWINR_PRINTER_HEADER_TEXT": "TWINR.com",
                "TWINR_PRINTER_LINE_WIDTH": "28",
                "TWINR_PRINTER_FEED_LINES": "3",
                "TWINR_PRINTER_QUEUE": "Thermal_GP58",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("OPENAI_MODEL=gpt-4o-mini", env_text)
        self.assertIn("TWINR_WEB_PORT=1440", env_text)
        self.assertIn("TWINR_PRINTER_LINE_WIDTH=28", env_text)

    def test_settings_page_renders_extended_sections_and_hover_help(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/settings")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Models and voices", response.text)
        self.assertIn("Search", response.text)
        self.assertIn("Camera and vision", response.text)
        self.assertIn("Proactive behavior", response.text)
        self.assertIn("Buttons and motion sensor", response.text)
        self.assertIn("Display and printer", response.text)
        self.assertIn("field-tooltip", response.text)
        self.assertIn("How much image detail Twinr asks OpenAI to inspect.", response.text)
        self.assertIn("Optional speaking instructions sent with text-to-speech requests.", response.text)

    def test_settings_post_updates_extended_env_values(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/settings",
            data={
                "OPENAI_REASONING_EFFORT": "high",
                "OPENAI_TTS_FORMAT": "mp3",
                "OPENAI_TTS_INSTRUCTIONS": "Speak warm, clear German.",
                "OPENAI_REALTIME_TRANSCRIPTION_MODEL": "whisper-1",
                "OPENAI_REALTIME_LANGUAGE": "de",
                "OPENAI_REALTIME_INPUT_SAMPLE_RATE": "16000",
                "OPENAI_REALTIME_INSTRUCTIONS": "Stay calm and concise.",
                "TWINR_OPENAI_ENABLE_WEB_SEARCH": "true",
                "TWINR_CONVERSATION_WEB_SEARCH": "always",
                "OPENAI_SEARCH_MODEL": "gpt-5.2-chat-latest",
                "TWINR_OPENAI_WEB_SEARCH_CONTEXT_SIZE": "high",
                "TWINR_OPENAI_WEB_SEARCH_COUNTRY": "DE",
                "TWINR_AUDIO_INPUT_DEVICE": "plughw:2,0",
                "TWINR_AUDIO_OUTPUT_DEVICE": "default",
                "TWINR_AUDIO_SAMPLE_RATE": "22050",
                "TWINR_AUDIO_CHANNELS": "1",
                "TWINR_AUDIO_CHUNK_MS": "120",
                "TWINR_AUDIO_PREROLL_MS": "450",
                "TWINR_AUDIO_SPEECH_START_CHUNKS": "2",
                "TWINR_AUDIO_START_TIMEOUT_S": "6.5",
                "TWINR_AUDIO_MAX_RECORD_SECONDS": "25",
                "TWINR_AUDIO_BEEP_VOLUME": "0.65",
                "TWINR_SEARCH_FEEDBACK_TONES_ENABLED": "false",
                "TWINR_CAMERA_DEVICE": "/dev/video2",
                "TWINR_CAMERA_WIDTH": "800",
                "TWINR_CAMERA_HEIGHT": "600",
                "TWINR_CAMERA_FRAMERATE": "25",
                "TWINR_CAMERA_INPUT_FORMAT": "mjpeg",
                "TWINR_CAMERA_FFMPEG_PATH": "/usr/bin/ffmpeg",
                "OPENAI_VISION_DETAIL": "high",
                "TWINR_VISION_REFERENCE_IMAGE": "/home/thh/reference-user.jpg",
                "TWINR_USER_DISPLAY_NAME": "Thom",
                "TWINR_PROACTIVE_ENABLED": "true",
                "TWINR_PROACTIVE_POLL_INTERVAL_S": "3.5",
                "TWINR_PROACTIVE_AUDIO_ENABLED": "true",
                "TWINR_PROACTIVE_AUDIO_DEVICE": "plughw:CARD=CameraB409241,DEV=0",
                "TWINR_PROACTIVE_AUDIO_SAMPLE_MS": "900",
                "TWINR_WEB_HOST": "127.0.0.1",
                "TWINR_WEB_PORT": "1441",
                "TWINR_PERSONALITY_DIR": "personality",
                "TWINR_RUNTIME_STATE_PATH": "/tmp/twinr-runtime-state.json",
                "TWINR_MEMORY_MARKDOWN_PATH": "/tmp/MEMORY.md",
                "TWINR_REMINDER_STORE_PATH": "/tmp/reminders.json",
                "TWINR_RESTORE_RUNTIME_STATE_ON_STARTUP": "true",
                "TWINR_GPIO_CHIP": "gpiochip4",
                "TWINR_GREEN_BUTTON_GPIO": "23",
                "TWINR_YELLOW_BUTTON_GPIO": "22",
                "TWINR_BUTTON_ACTIVE_LOW": "true",
                "TWINR_BUTTON_BIAS": "pull-up",
                "TWINR_BUTTON_DEBOUNCE_MS": "90",
                "TWINR_BUTTON_PROBE_LINES": "23,22,24",
                "TWINR_PIR_MOTION_GPIO": "17",
                "TWINR_PIR_ACTIVE_HIGH": "true",
                "TWINR_PIR_BIAS": "pull-down",
                "TWINR_DISPLAY_DRIVER": "waveshare_4in2_v2",
                "TWINR_DISPLAY_VENDOR_DIR": "hardware/display/vendor",
                "TWINR_DISPLAY_SPI_BUS": "0",
                "TWINR_DISPLAY_SPI_DEVICE": "1",
                "TWINR_DISPLAY_CS_GPIO": "8",
                "TWINR_DISPLAY_DC_GPIO": "25",
                "TWINR_DISPLAY_RESET_GPIO": "17",
                "TWINR_DISPLAY_BUSY_GPIO": "24",
                "TWINR_DISPLAY_WIDTH": "400",
                "TWINR_DISPLAY_HEIGHT": "300",
                "TWINR_DISPLAY_ROTATION_DEGREES": "270",
                "TWINR_DISPLAY_FULL_REFRESH_INTERVAL": "2",
                "TWINR_DISPLAY_POLL_INTERVAL_S": "0.8",
                "TWINR_PRINTER_QUEUE": "Thermal_GP58",
                "TWINR_PRINTER_DEVICE_URI": "usb://Acme/Printer",
                "TWINR_PRINT_BUTTON_COOLDOWN_S": "3.5",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn("OPENAI_REASONING_EFFORT=high", env_text)
        self.assertIn('OPENAI_TTS_INSTRUCTIONS="Speak warm, clear German."', env_text)
        self.assertIn("TWINR_CONVERSATION_WEB_SEARCH=always", env_text)
        self.assertIn("TWINR_CAMERA_DEVICE=/dev/video2", env_text)
        self.assertIn("OPENAI_VISION_DETAIL=high", env_text)
        self.assertIn("TWINR_PROACTIVE_AUDIO_DEVICE=plughw:CARD=CameraB409241,DEV=0", env_text)
        self.assertIn("TWINR_RUNTIME_STATE_PATH=/tmp/twinr-runtime-state.json", env_text)
        self.assertIn("TWINR_BUTTON_PROBE_LINES=23,22,24", env_text)
        self.assertIn("TWINR_PRINTER_DEVICE_URI=usb://Acme/Printer", env_text)

    def test_memory_page_renders_live_snapshot(self) -> None:
        client, env_path = self.make_client()
        store = RuntimeSnapshotStore(env_path.parent / "runtime-state.json")
        PersistentMemoryMarkdownStore(env_path.parent / "state" / "MEMORY.md").remember(
            kind="appointment",
            summary="Arzttermin am Montag um 14 Uhr.",
            details="Bei Dr. Meyer in Hamburg.",
        )
        store.save(
            status="waiting",
            memory_turns=(
                ConversationTurn(
                    "system",
                    "Twinr memory summary:\n- Verified web lookup: Bus 24 -> 07:30 Uhr",
                    datetime(2026, 3, 12, 17, 59, tzinfo=timezone.utc),
                ),
                ConversationTurn(
                    "user",
                    "Erinnere mich an den Termin",
                    datetime(2026, 3, 12, 18, 0, tzinfo=timezone.utc),
                ),
                ConversationTurn(
                    "assistant",
                    "Der Termin ist um 14 Uhr.",
                    datetime(2026, 3, 12, 18, 0, 3, tzinfo=timezone.utc),
                ),
            ),
            memory_raw_tail=(
                ConversationTurn(
                    "user",
                    "Erinnere mich an den Termin",
                    datetime(2026, 3, 12, 18, 0, tzinfo=timezone.utc),
                ),
                ConversationTurn(
                    "assistant",
                    "Der Termin ist um 14 Uhr.",
                    datetime(2026, 3, 12, 18, 0, 3, tzinfo=timezone.utc),
                ),
            ),
            memory_ledger=(
                MemoryLedgerItem(
                    kind="conversation_summary",
                    content="User asked about an appointment. Twinr answered with the time.",
                    created_at=datetime(2026, 3, 12, 17, 58, tzinfo=timezone.utc),
                    source="compactor",
                ),
            ),
            memory_search_results=(
                SearchMemoryEntry(
                    question="Wann faehrt der Bus?",
                    answer="Bus 24 faehrt um 07:30 Uhr.",
                    sources=("https://example.com/fahrplan",),
                    created_at=datetime(2026, 3, 12, 17, 57, tzinfo=timezone.utc),
                    location_hint="Schwarzenbek",
                    date_context="2026-03-13",
                ),
            ),
            memory_state=MemoryState(
                active_topic="Termin",
                last_user_goal="Termindetails behalten",
                pending_printable="Der Termin ist um 14 Uhr.",
                last_search_summary="Bus 24 faehrt um 07:30 Uhr.",
                open_loops=("Termin bestaetigen",),
            ),
            last_transcript="Erinnere mich an den Termin",
            last_response="Der Termin ist um 14 Uhr.",
        )

        response = client.get("/memory")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live memory snapshot", response.text)
        self.assertIn("Durable memories", response.text)
        self.assertIn("Memory state", response.text)
        self.assertIn("Raw tail", response.text)
        self.assertIn("Recent search results", response.text)
        self.assertIn("Arzttermin am Montag um 14 Uhr.", response.text)
        self.assertIn("Erinnere mich an den Termin", response.text)
        self.assertIn("Der Termin ist um 14 Uhr.", response.text)
        self.assertIn("Bus 24 faehrt um 07:30 Uhr.", response.text)
        self.assertIn("Termindetails behalten", response.text)

    def test_memory_post_adds_durable_memory_entry(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/memory",
            data={
                "_action": "add_memory",
                "memory_kind": "appointment",
                "memory_summary": "Arzttermin am Montag um 14 Uhr.",
                "memory_details": "Bei Dr. Meyer in Hamburg.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        memory_entries = PersistentMemoryMarkdownStore(env_path.parent / "state" / "MEMORY.md").load_entries()
        self.assertEqual(len(memory_entries), 1)
        self.assertEqual(memory_entries[0].kind, "appointment")
        self.assertEqual(memory_entries[0].summary, "Arzttermin am Montag um 14 Uhr.")

    def test_personality_post_updates_base_files_and_preserves_managed_section(self) -> None:
        client, env_path = self.make_client()
        personality_dir = env_path.parent / "personality"
        ManagedContextFileStore(
            personality_dir / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        ).upsert(category="humor", instruction="Use only light humor.")

        response = client.post(
            "/personality",
            data={
                "SYSTEM": "Updated system",
                "PERSONALITY_BASE": "Updated personality",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual((personality_dir / "SYSTEM.md").read_text(encoding="utf-8"), "Updated system\n")
        personality_text = (personality_dir / "PERSONALITY.md").read_text(encoding="utf-8")
        self.assertIn("Updated personality", personality_text)
        self.assertIn("humor: Use only light humor.", personality_text)

    def test_personality_post_adds_managed_update(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/personality",
            data={
                "_action": "upsert_managed",
                "category": "response_style",
                "instruction": "Keep answers short and calm.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ManagedContextFileStore(
            env_path.parent / "personality" / "PERSONALITY.md",
            section_title="Twinr managed personality updates",
        ).load_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "response_style")
        self.assertEqual(entries[0].instruction, "Keep answers short and calm.")

    def test_user_post_adds_managed_profile_update(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/user",
            data={
                "_action": "upsert_managed",
                "category": "pets",
                "instruction": "Thom has two dogs.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ManagedContextFileStore(
            env_path.parent / "personality" / "USER.md",
            section_title="Twinr managed user updates",
        ).load_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, "pets")
        self.assertEqual(entries[0].instruction, "Thom has two dogs.")

    def test_ops_logs_page_renders_structured_events(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        TwinrOpsEventStore.from_config(config).append(
            event="turn_started",
            message="Green button started a conversation turn.",
            data={"button": "green"},
        )

        response = client.get("/ops/logs")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Latest structured events", response.text)
        self.assertIn("turn_started", response.text)
        self.assertIn("Green button started a conversation turn.", response.text)

    def test_ops_usage_page_renders_usage_summary(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        from twinr.ops import TokenUsage, TwinrUsageStore

        TwinrUsageStore.from_config(config).append(
            source="hardware_loop",
            request_kind="conversation",
            model="gpt-5.2",
            response_id="resp_usage_1",
            token_usage=TokenUsage(input_tokens=90, output_tokens=30, total_tokens=120),
        )

        response = client.get("/ops/usage")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Usage summary", response.text)
        self.assertIn("resp_usage_1", response.text)
        self.assertIn("gpt-5.2", response.text)

    def test_ops_health_page_renders_system_health(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/health")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live system health", response.text)
        self.assertIn("Services", response.text)

    def test_ops_self_test_page_lists_pir_motion_and_proactive_mic(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/self-test")

        self.assertEqual(response.status_code, 200)
        self.assertIn("PIR motion", response.text)
        self.assertIn("Wait for a motion trigger on the configured PIR input.", response.text)
        self.assertIn("Proaktives Mikrofon", response.text)
        self.assertIn("proactive background microphone", response.text)

    def test_ops_support_post_builds_bundle(self) -> None:
        client, env_path = self.make_client()

        response = client.post("/ops/support")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Latest bundle", response.text)
        bundle_files = list(resolve_ops_paths(env_path.parent).bundles_root.glob("*.zip"))
        self.assertEqual(len(bundle_files), 1)

    def test_ops_self_test_route_renders_runner_result(self) -> None:
        client, _env_path = self.make_client()
        fake_result = SimpleNamespace(
            status="ok",
            summary="Confirmation beep played.",
            details=("Output device: default",),
            artifact_name="speaker-test.txt",
            finished_at="2026-03-13T08:00:00+00:00",
        )

        with patch("twinr.web.app.TwinrSelfTestRunner") as runner_cls:
            runner_cls.available_tests.return_value = (
                ("speaker", "Speaker-Beep", "Play a local confirmation beep."),
            )
            runner_cls.return_value.run.return_value = fake_result

            response = client.post("/ops/self-test", data={"test_name": "speaker"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("Last result", response.text)
        self.assertIn("Confirmation beep played.", response.text)
        self.assertIn("speaker-test.txt", response.text)


if __name__ == "__main__":
    unittest.main()
