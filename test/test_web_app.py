from array import array
from datetime import datetime, timezone
import io
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from twinr.agent.base_agent import AdaptiveTimingStore, RuntimeSnapshotStore, TwinrConfig
from twinr.automations import AutomationAction, AutomationStore, build_sensor_trigger
from twinr.memory.context_store import ManagedContextFileStore, PersistentMemoryMarkdownStore
from twinr.memory.reminders import ReminderStore
from twinr.memory import ConversationTurn, MemoryLedgerItem, MemoryState, SearchMemoryEntry
from twinr.integrations import ManagedIntegrationConfig, TwinrIntegrationStore
from twinr.ops import DeviceFact, DeviceOverview, DeviceStatus, TwinrOpsEventStore, resolve_ops_paths
from twinr.web import create_app


def _voice_sample_wav_bytes(*, frequency_hz: float = 175.0, amplitude: float = 0.35, duration_s: float = 1.8) -> bytes:
    sample_rate = 16000
    total_frames = int(sample_rate * duration_s)
    frames = array("h")
    for index in range(total_frames):
        t = index / sample_rate
        envelope = min(1.0, index / (sample_rate * 0.2), (total_frames - index) / (sample_rate * 0.2))
        sample = amplitude * envelope * (
            (0.70 * math.sin(2.0 * math.pi * frequency_hz * t))
            + (0.20 * math.sin(2.0 * math.pi * frequency_hz * 2.0 * t))
            + (0.10 * math.sin(2.0 * math.pi * (frequency_hz + 35.0) * t))
        )
        frames.append(max(-32767, min(32767, int(sample * 32767))))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())
    return buffer.getvalue()


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
        self.assertIn("Reminders", response.text)
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

    def test_integrations_page_renders_mail_and_calendar_forms(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/integrations")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Integration overview", response.text)
        self.assertIn("Save email integration", response.text)
        self.assertIn("Save calendar integration", response.text)
        self.assertIn("Gmail", response.text)
        self.assertIn("ICS file", response.text)

    def test_integrations_post_saves_email_config_and_secret(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "email_mailbox",
                "enabled": "true",
                "profile": "gmail",
                "account_email": "anna@gmail.com",
                "from_address": "anna@gmail.com",
                "TWINR_INTEGRATION_EMAIL_APP_PASSWORD": "abcd efgh ijkl mnop",
                "imap_host": "",
                "imap_port": "",
                "imap_mailbox": "INBOX",
                "smtp_host": "",
                "smtp_port": "",
                "unread_only_default": "true",
                "restrict_reads_to_known_senders": "false",
                "restrict_recipients_to_known_contacts": "false",
                "known_contacts_text": "Anna <anna@gmail.com>",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        env_text = env_path.read_text(encoding="utf-8")
        self.assertIn('TWINR_INTEGRATION_EMAIL_APP_PASSWORD="abcd efgh ijkl mnop"', env_text)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("email_mailbox")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("imap_host"), "imap.gmail.com")
        self.assertEqual(record.value("smtp_host"), "smtp.gmail.com")
        self.assertEqual(record.value("account_email"), "anna@gmail.com")
        store_text = TwinrIntegrationStore.from_project_root(env_path.parent).path.read_text(encoding="utf-8")
        self.assertNotIn("abcd efgh ijkl mnop", store_text)

        response = client.get("/integrations")
        self.assertNotIn("abcd", response.text)
        self.assertNotIn("mnop", response.text)
        self.assertIn("Credential state: Configured.", response.text)
        self.assertIn("credential stored separately in .env", response.text)

    def test_integrations_post_saves_calendar_config(self) -> None:
        client, env_path = self.make_client()
        calendar_path = env_path.parent / "state" / "calendar.ics"
        calendar_path.parent.mkdir(parents=True, exist_ok=True)
        calendar_path.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "calendar_agenda",
                "enabled": "true",
                "source_kind": "ics_file",
                "source_value": "state/calendar.ics",
                "timezone": "Europe/Berlin",
                "default_upcoming_days": "5",
                "max_events": "10",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("calendar_agenda")
        self.assertTrue(record.enabled)
        self.assertEqual(record.value("source_value"), "state/calendar.ics")
        response = client.get("/integrations")
        self.assertIn("Ready", response.text)
        self.assertIn("state/calendar.ics", response.text)

    def test_integrations_post_rejects_calendar_url_with_query_token(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/integrations",
            data={
                "_integration_id": "calendar_agenda",
                "enabled": "true",
                "source_kind": "ics_url",
                "source_value": "https://calendar.example.com/feed.ics?token=super-secret",
                "timezone": "Europe/Berlin",
                "default_upcoming_days": "5",
                "max_events": "10",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("error=", response.headers["location"])
        record = TwinrIntegrationStore.from_project_root(env_path.parent).get("calendar_agenda")
        self.assertFalse(record.enabled)
        if TwinrIntegrationStore.from_project_root(env_path.parent).path.exists():
            store_text = TwinrIntegrationStore.from_project_root(env_path.parent).path.read_text(encoding="utf-8")
            self.assertNotIn("super-secret", store_text)

    def test_voice_profile_page_renders_status_and_actions(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/voice-profile")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live voice status", response.text)
        self.assertIn("Capture and enroll sample", response.text)
        self.assertIn("Verify now", response.text)
        self.assertIn("Reset profile", response.text)
        self.assertIn("No raw enrollment audio is stored", response.text)

    def test_voice_profile_post_enroll_verify_and_reset(self) -> None:
        client, env_path = self.make_client()
        sample = _voice_sample_wav_bytes()
        voice_store_path = env_path.parent / "state" / "voice_profile.json"

        with patch("twinr.web.app._capture_voice_profile_sample", return_value=sample):
            enroll_response = client.post("/voice-profile", data={"_action": "enroll"})
        self.assertTrue(voice_store_path.exists())

        with patch("twinr.web.app._capture_voice_profile_sample", return_value=sample):
            verify_response = client.post("/voice-profile", data={"_action": "verify"})
        reset_response = client.post("/voice-profile", data={"_action": "reset"})

        self.assertEqual(enroll_response.status_code, 200)
        self.assertIn("Profile updated", enroll_response.text)
        self.assertIn("No raw audio was kept.", enroll_response.text)

        self.assertEqual(verify_response.status_code, 200)
        self.assertIn("Likely user", verify_response.text)
        self.assertIn("Confidence", verify_response.text)

        self.assertEqual(reset_response.status_code, 200)
        self.assertIn("Profile reset", reset_response.text)
        self.assertFalse(voice_store_path.exists())

    def test_automations_page_renders_family_sections_and_forms(self) -> None:
        client, env_path = self.make_client()
        store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        store.create_time_automation(
            name="Morning weather",
            description="Speak the weather every morning.",
            schedule="daily",
            time_of_day="08:00",
            timezone_name="Europe/Berlin",
            actions=(
                AutomationAction(
                    kind="llm_prompt",
                    text="Tell Thom the morning weather in Schwarzenbek.",
                    payload={"delivery": "spoken", "allow_web_search": True},
                ),
            ),
        )
        sensor_trigger = build_sensor_trigger("vad_quiet", hold_seconds=30, cooldown_seconds=180)
        store.create_if_then_automation(
            name="Quiet room check-in",
            description="Offer help if the room stays quiet.",
            event_name=sensor_trigger.event_name,
            all_conditions=sensor_trigger.all_conditions,
            any_conditions=sensor_trigger.any_conditions,
            cooldown_seconds=sensor_trigger.cooldown_seconds,
            actions=(AutomationAction(kind="say", text="Ich bin weiter hier, falls du etwas brauchst."),),
        )

        response = client.get("/automations")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Automation families", response.text)
        self.assertIn("Scheduled", response.text)
        self.assertIn("Sensor-triggered", response.text)
        self.assertIn("Email Mailbox automations", response.text)
        self.assertIn("Calendar Agenda automations", response.text)
        self.assertIn("Integration not configured", response.text)
        self.assertIn("Morning weather", response.text)
        self.assertIn("Quiet room check-in", response.text)
        self.assertIn("Add scheduled automation", response.text)
        self.assertIn("Add sensor automation", response.text)
        self.assertIn("Tell Thom the morning weather in Schwarzenbek.", response.text)
        self.assertIn("room microphone has been quiet", response.text)

    def test_automations_page_shows_configured_integration_family_state(self) -> None:
        client, env_path = self.make_client()
        TwinrIntegrationStore.from_project_root(env_path.parent).save(
            ManagedIntegrationConfig(
                integration_id="email_mailbox",
                enabled=True,
                settings={"account_email": "anna@example.com"},
            )
        )

        response = client.get("/automations")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Email Mailbox automations", response.text)
        self.assertIn("Integration configured", response.text)

    def test_automations_post_creates_time_automation(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/automations",
            data={
                "_action": "save_time_automation",
                "automation_id": "",
                "name": "Daily headlines",
                "description": "Print the top headlines each morning.",
                "enabled": "true",
                "schedule": "daily",
                "due_at": "",
                "time_of_day": "08:00",
                "weekdays_text": "",
                "timezone_name": "Europe/Berlin",
                "tags_text": "news, morning",
                "delivery": "printed",
                "content_mode": "llm_prompt",
                "allow_web_search": "true",
                "content": "Print the top headlines of the day in short German bullet points.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").load_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.name, "Daily headlines")
        self.assertEqual(entry.trigger.schedule, "daily")
        self.assertEqual(entry.trigger.time_of_day, "08:00")
        self.assertEqual(entry.actions[0].kind, "llm_prompt")
        self.assertEqual(entry.actions[0].payload["delivery"], "printed")
        self.assertTrue(entry.actions[0].payload["allow_web_search"])
        self.assertEqual(entry.tags, ("news", "morning"))

    def test_automations_post_creates_sensor_automation(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/automations",
            data={
                "_action": "save_sensor_automation",
                "automation_id": "",
                "name": "Welcome after motion",
                "description": "Greet after motion is seen.",
                "enabled": "true",
                "trigger_kind": "pir_motion_detected",
                "hold_seconds": "",
                "cooldown_seconds": "120",
                "tags_text": "sensor, welcome",
                "delivery": "spoken",
                "content_mode": "static_text",
                "allow_web_search": "false",
                "content": "Hallo, ich bin bereit.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").load_entries()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.name, "Welcome after motion")
        self.assertEqual(entry.actions[0].kind, "say")
        self.assertEqual(entry.actions[0].text, "Hallo, ich bin bereit.")
        self.assertEqual(entry.trigger.event_name, "pir.motion_detected")
        self.assertEqual(entry.tags, ("sensor", "welcome"))

    def test_automations_post_toggles_and_deletes_automation(self) -> None:
        client, env_path = self.make_client()
        store = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin")
        entry = store.create_time_automation(
            name="Morning greeting",
            schedule="daily",
            time_of_day="09:00",
            timezone_name="Europe/Berlin",
            actions=(AutomationAction(kind="say", text="Guten Morgen."),),
        )

        toggle_response = client.post(
            "/automations",
            data={"_action": "toggle_automation", "automation_id": entry.automation_id},
            follow_redirects=False,
        )
        self.assertEqual(toggle_response.status_code, 303)
        toggled = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").get(entry.automation_id)
        self.assertIsNotNone(toggled)
        self.assertFalse(toggled.enabled)

        delete_response = client.post(
            "/automations",
            data={"_action": "delete_automation", "automation_id": entry.automation_id},
            follow_redirects=False,
        )
        self.assertEqual(delete_response.status_code, 303)
        remaining = AutomationStore(env_path.parent / "state" / "automations.json", timezone_name="Europe/Berlin").load_entries()
        self.assertEqual(remaining, ())

    def test_settings_post_updates_env(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/settings",
            data={
                "OPENAI_MODEL": "gpt-4o-mini",
                "OPENAI_STT_MODEL": "whisper-1",
                "OPENAI_TTS_MODEL": "gpt-4o-mini-tts",
                "OPENAI_TTS_VOICE": "marin",
                "OPENAI_TTS_SPEED": "0.90",
                "OPENAI_REALTIME_MODEL": "gpt-4o-realtime-preview",
                "OPENAI_REALTIME_VOICE": "sage",
                "OPENAI_REALTIME_SPEED": "1.05",
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
        self.assertIn("OPENAI_TTS_SPEED=0.90", env_text)
        self.assertIn("OPENAI_REALTIME_SPEED=1.05", env_text)
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
        self.assertIn("Proactive timing", response.text)
        self.assertIn("Proactive sensitivity", response.text)
        self.assertIn("Wakeword mode", response.text)
        self.assertIn("Primary wakeword detector", response.text)
        self.assertIn("Fallback detector", response.text)
        self.assertIn("STT verifier", response.text)
        self.assertIn("Wakeword phrases", response.text)
        self.assertIn("openWakeWord models", response.text)
        self.assertIn("Calibration profile", response.text)
        self.assertIn("Recommended profile", response.text)
        self.assertIn("Wake presence grace (s)", response.text)
        self.assertIn("Wake audio ratio", response.text)
        self.assertIn("Wake active chunks", response.text)
        self.assertIn("Wake patience frames", response.text)
        self.assertIn("Wakeword min score", response.text)
        self.assertIn("Buttons and motion sensor", response.text)
        self.assertIn("Display and printer", response.text)
        self.assertIn("Adaptive timing", response.text)
        self.assertIn("Observed patterns", response.text)
        self.assertIn("Configured baselines", response.text)
        self.assertIn("Reset learned timing", response.text)
        self.assertIn("field-tooltip", response.text)
        self.assertIn("How much image detail Twinr asks OpenAI to inspect.", response.text)
        self.assertIn("Optional speaking instructions sent with text-to-speech requests.", response.text)
        self.assertIn("TTS speed", response.text)
        self.assertIn("Realtime speed", response.text)
        self.assertIn(
            "openWakeWord is the professional default for passive listening. STT stays available as an explicit degraded fallback path.",
            response.text,
        )
        self.assertIn("After this many quiet seconds without motion, the scene is treated as idle / low-motion.", response.text)

    def test_settings_page_renders_current_adaptive_timing_profile(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        store = AdaptiveTimingStore(config.adaptive_timing_store_path, config=config)
        store.record_no_speech_timeout(initial_source="button", follow_up=False)
        store.record_capture(
            initial_source="button",
            follow_up=False,
            speech_started_after_ms=1800,
            resumed_after_pause_count=1,
        )

        response = client.get("/settings")

        self.assertEqual(response.status_code, 200)
        self.assertIn("8.75 s", response.text)
        self.assertIn("1230 ms", response.text)
        self.assertIn("470 ms", response.text)
        self.assertIn("1 ok / 1 timeout", response.text)
        self.assertIn("Persistent learned timing profile on disk.", response.text)

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
                "TWINR_PROACTIVE_CAPTURE_INTERVAL_S": "7.0",
                "TWINR_PROACTIVE_LOW_MOTION_AFTER_S": "14.0",
                "TWINR_PROACTIVE_AUDIO_ENABLED": "true",
                "TWINR_PROACTIVE_AUDIO_DEVICE": "plughw:CARD=CameraB409241,DEV=0",
                "TWINR_PROACTIVE_AUDIO_SAMPLE_MS": "900",
                "TWINR_WAKEWORD_ENABLED": "true",
                "TWINR_WAKEWORD_PHRASES": "hey twinr, hey twinna, twinr",
                "TWINR_WAKEWORD_SAMPLE_MS": "1700",
                "TWINR_WAKEWORD_PRESENCE_GRACE_S": "600",
                "TWINR_WAKEWORD_MOTION_GRACE_S": "180",
                "TWINR_WAKEWORD_SPEECH_GRACE_S": "75",
                "TWINR_WAKEWORD_ATTEMPT_COOLDOWN_S": "5.5",
                "TWINR_WAKEWORD_MIN_ACTIVE_RATIO": "0.06",
                "TWINR_WAKEWORD_MIN_ACTIVE_CHUNKS": "2",
                "TWINR_PROACTIVE_ATTENTION_WINDOW_S": "8.5",
                "TWINR_PROACTIVE_FLOOR_STILLNESS_S": "26.0",
                "TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD": "0.76",
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
        self.assertIn("TWINR_PROACTIVE_LOW_MOTION_AFTER_S=14.0", env_text)
        self.assertIn("TWINR_WAKEWORD_ENABLED=true", env_text)
        self.assertIn('TWINR_WAKEWORD_PHRASES="hey twinr, hey twinna, twinr"', env_text)
        self.assertIn("TWINR_WAKEWORD_SAMPLE_MS=1700", env_text)
        self.assertIn("TWINR_WAKEWORD_MIN_ACTIVE_RATIO=0.06", env_text)
        self.assertIn("TWINR_WAKEWORD_MIN_ACTIVE_CHUNKS=2", env_text)
        self.assertIn("TWINR_PROACTIVE_ATTENTION_WINDOW_S=8.5", env_text)
        self.assertIn("TWINR_PROACTIVE_SHOWING_INTENT_SCORE_THRESHOLD=0.76", env_text)
        self.assertIn("TWINR_RUNTIME_STATE_PATH=/tmp/twinr-runtime-state.json", env_text)
        self.assertIn("TWINR_BUTTON_PROBE_LINES=23,22,24", env_text)
        self.assertIn("TWINR_PRINTER_DEVICE_URI=usb://Acme/Printer", env_text)

    def test_settings_post_resets_adaptive_timing_profile(self) -> None:
        client, env_path = self.make_client()
        config = TwinrConfig.from_env(env_path)
        store = AdaptiveTimingStore(config.adaptive_timing_store_path, config=config)
        store.record_no_speech_timeout(initial_source="button", follow_up=False)
        store.record_capture(
            initial_source="button",
            follow_up=False,
            speech_started_after_ms=2200,
            resumed_after_pause_count=1,
        )

        response = client.post(
            "/settings",
            data={"_action": "reset_adaptive_timing"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/settings?saved=1")
        reset_profile = store.current()
        default_profile = store.default_profile()
        self.assertEqual(reset_profile.button_start_timeout_s, default_profile.button_start_timeout_s)
        self.assertEqual(reset_profile.follow_up_start_timeout_s, default_profile.follow_up_start_timeout_s)
        self.assertEqual(reset_profile.speech_pause_ms, default_profile.speech_pause_ms)
        self.assertEqual(reset_profile.pause_grace_ms, default_profile.pause_grace_ms)
        self.assertEqual(reset_profile.button_success_count, 0)
        self.assertEqual(reset_profile.button_timeout_count, 0)
        self.assertEqual(reset_profile.pause_resume_count, 0)

    def test_memory_page_renders_live_snapshot(self) -> None:
        client, env_path = self.make_client()
        store = RuntimeSnapshotStore(env_path.parent / "runtime-state.json")
        PersistentMemoryMarkdownStore(env_path.parent / "state" / "MEMORY.md").remember(
            kind="appointment",
            summary="Arzttermin am Montag um 14 Uhr.",
            details="Bei Dr. Meyer in Hamburg.",
        )
        reminder_store = ReminderStore(env_path.parent / "state" / "reminders.json")
        reminder_store.schedule(
            due_at="2030-03-15T09:00",
            kind="medication",
            summary="An die Tabletten erinnern.",
            details="Nach dem Fruehstueck.",
        )
        delivered = reminder_store.schedule(
            due_at="2030-03-15T12:00",
            kind="appointment",
            summary="An den Arzttermin erinnern.",
            details="Dr. Meyer in Hamburg.",
        )
        reminder_store.mark_delivered(delivered.reminder_id, delivered_at=datetime(2026, 3, 13, 12, 5, tzinfo=timezone.utc))
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
        self.assertIn("Scheduled reminders", response.text)
        self.assertIn("Pending reminders", response.text)
        self.assertIn("Delivered reminders", response.text)
        self.assertIn("An die Tabletten erinnern.", response.text)
        self.assertIn("An den Arzttermin erinnern.", response.text)
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

    def test_memory_post_adds_reminder_entry(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/memory",
            data={
                "_action": "add_reminder",
                "reminder_due_at": "2030-03-15T12:00",
                "reminder_kind": "appointment",
                "reminder_summary": "An den Arzttermin erinnern.",
                "reminder_details": "Unterlagen mitnehmen.",
                "reminder_original_request": "Erinnere mich morgen um 12 Uhr an den Arzttermin.",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        reminder_entries = ReminderStore(env_path.parent / "state" / "reminders.json").load_entries()
        self.assertEqual(len(reminder_entries), 1)
        self.assertEqual(reminder_entries[0].kind, "appointment")
        self.assertEqual(reminder_entries[0].summary, "An den Arzttermin erinnern.")

    def test_memory_post_marks_reminder_delivered(self) -> None:
        client, env_path = self.make_client()
        reminder_store = ReminderStore(env_path.parent / "state" / "reminders.json")
        reminder = reminder_store.schedule(
            due_at="2030-03-15T12:00",
            kind="appointment",
            summary="An den Arzttermin erinnern.",
        )

        response = client.post(
            "/memory",
            data={
                "_action": "mark_reminder_delivered",
                "reminder_id": reminder.reminder_id,
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ReminderStore(env_path.parent / "state" / "reminders.json").load_entries()
        self.assertTrue(entries[0].delivered)

    def test_memory_post_deletes_reminder(self) -> None:
        client, env_path = self.make_client()
        reminder_store = ReminderStore(env_path.parent / "state" / "reminders.json")
        reminder = reminder_store.schedule(
            due_at="2030-03-15T12:00",
            kind="appointment",
            summary="An den Arzttermin erinnern.",
        )

        response = client.post(
            "/memory",
            data={
                "_action": "delete_reminder",
                "reminder_id": reminder.reminder_id,
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        entries = ReminderStore(env_path.parent / "state" / "reminders.json").load_entries()
        self.assertEqual(entries, ())

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

    def test_ops_devices_page_renders_device_status(self) -> None:
        client, _env_path = self.make_client()
        fake_overview = DeviceOverview(
            captured_at="2026-03-13T16:10:00+00:00",
            devices=(
                DeviceStatus(
                    key="printer",
                    label="Printer",
                    status="warn",
                    summary="Queue is visible, but paper output must be confirmed on the device.",
                    facts=(
                        DeviceFact("Queue", "Thermal_GP58"),
                        DeviceFact("Paper status", "unknown on the current raw USB/CUPS path"),
                    ),
                    notes=("Twinr cannot prove real paper output from this printer path.",),
                ),
                DeviceStatus(
                    key="camera",
                    label="Camera",
                    status="ok",
                    summary="Camera device `/dev/video0` is present.",
                    facts=(DeviceFact("Device", "/dev/video0"),),
                ),
            ),
        )

        with patch("twinr.web.app.collect_device_overview", return_value=fake_overview):
            response = client.get("/ops/devices")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Hardware devices", response.text)
        self.assertIn("Queue is visible, but paper output must be confirmed on the device.", response.text)
        self.assertIn("Paper status", response.text)
        self.assertIn("unknown on the current raw USB/CUPS path", response.text)
        self.assertIn("Camera device `/dev/video0` is present.", response.text)

    def test_ops_self_test_page_lists_pir_motion_and_proactive_mic(self) -> None:
        client, _env_path = self.make_client()

        response = client.get("/ops/self-test")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Printer-Testdruck", response.text)
        self.assertIn("confirm the paper output on the device", response.text)
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
