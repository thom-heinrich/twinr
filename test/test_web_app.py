from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from twinr.agent.base_agent.runtime_state import RuntimeSnapshotStore
from twinr.memory import ConversationTurn
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

    def test_memory_page_renders_live_snapshot(self) -> None:
        client, env_path = self.make_client()
        store = RuntimeSnapshotStore(env_path.parent / "runtime-state.json")
        store.save(
            status="waiting",
            memory_turns=(
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
            last_transcript="Erinnere mich an den Termin",
            last_response="Der Termin ist um 14 Uhr.",
        )

        response = client.get("/memory")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Live memory snapshot", response.text)
        self.assertIn("Erinnere mich an den Termin", response.text)
        self.assertIn("Der Termin ist um 14 Uhr.", response.text)

    def test_personality_post_updates_markdown_files(self) -> None:
        client, env_path = self.make_client()

        response = client.post(
            "/personality",
            data={
                "SYSTEM": "Updated system",
                "PERSONALITY": "Updated personality",
            },
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        personality_dir = env_path.parent / "personality"
        self.assertEqual((personality_dir / "SYSTEM.md").read_text(encoding="utf-8"), "Updated system\n")
        self.assertEqual(
            (personality_dir / "PERSONALITY.md").read_text(encoding="utf-8"),
            "Updated personality\n",
        )


if __name__ == "__main__":
    unittest.main()
