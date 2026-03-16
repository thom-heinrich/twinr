from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.settings.simple_settings import (
    adjustable_settings_context,
    current_speech_speed,
    current_spoken_voice,
    spoken_voice_language_note,
    memory_capacity_level,
    supported_spoken_voices,
    update_simple_setting,
    write_env_updates,
)
from twinr.config import TwinrConfig


class SimpleSettingsTests(unittest.TestCase):
    def test_memory_capacity_increase_updates_both_memory_values(self) -> None:
        config = TwinrConfig(memory_max_turns=20, memory_keep_recent=10)

        result = update_simple_setting(config, setting="memory_capacity", action="increase")

        self.assertTrue(result.changed)
        self.assertEqual(result.env_updates["TWINR_MEMORY_MAX_TURNS"], "28")
        self.assertEqual(result.env_updates["TWINR_MEMORY_KEEP_RECENT"], "12")
        self.assertEqual(result.data["level"], 3)

    def test_numeric_settings_are_clamped_to_safe_bounds(self) -> None:
        config = TwinrConfig(speech_pause_ms=2000, conversation_follow_up_timeout_s=7.5)

        pause = update_simple_setting(config, setting="speech_pause_ms", action="increase")
        follow_up = update_simple_setting(config, setting="follow_up_timeout_s", action="set", value=99)

        self.assertEqual(pause.env_updates["TWINR_SPEECH_PAUSE_MS"], "2200")
        self.assertEqual(follow_up.env_updates["TWINR_CONVERSATION_FOLLOW_UP_TIMEOUT_S"], "8.0")

    def test_spoken_voice_updates_both_tts_and_realtime_voice(self) -> None:
        config = TwinrConfig(openai_tts_voice="marin", openai_realtime_voice="sage")

        result = update_simple_setting(config, setting="spoken_voice", action="set", value="cedar")

        self.assertTrue(result.changed)
        self.assertEqual(result.env_updates["OPENAI_TTS_VOICE"], "cedar")
        self.assertEqual(result.env_updates["OPENAI_REALTIME_VOICE"], "cedar")
        self.assertEqual(result.data["voice"], "cedar")

    def test_spoken_voice_requires_supported_catalog_value(self) -> None:
        config = TwinrConfig(openai_tts_voice="marin", openai_realtime_voice="sage")

        with self.assertRaises(ValueError):
            update_simple_setting(
                config,
                setting="spoken_voice",
                action="set",
                value="eine männliche Stimme",
            )

    def test_speech_speed_clamps_and_updates_both_audio_paths(self) -> None:
        config = TwinrConfig(openai_tts_speed=1.0, openai_realtime_speed=1.0)

        result = update_simple_setting(config, setting="speech_speed", action="set", value=2.5)

        self.assertTrue(result.changed)
        self.assertEqual(result.env_updates["OPENAI_TTS_SPEED"], "1.15")
        self.assertEqual(result.env_updates["OPENAI_REALTIME_SPEED"], "1.15")
        self.assertEqual(result.data["speech_speed"], 1.15)

    def test_adjustable_settings_context_reports_current_values(self) -> None:
        config = TwinrConfig(
            memory_max_turns=28,
            memory_keep_recent=12,
            openai_tts_voice="cedar",
            openai_realtime_voice="cedar",
            openai_tts_speed=0.9,
            openai_realtime_speed=0.9,
            speech_pause_ms=1400,
        )

        context = adjustable_settings_context(config)

        self.assertIn("memory_capacity level 3/4", context)
        self.assertIn("spoken_voice cedar", context)
        self.assertIn("speech_speed 0.90x", context)
        self.assertIn("All supported Twinr spoken voices can speak German.", context)
        self.assertIn("speech_pause_ms 1400", context)
        self.assertIn("follow_up_timeout_s 4.0", context)

    def test_write_env_updates_preserves_comments_and_updates_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text("# existing\nTWINR_MEMORY_MAX_TURNS=20\n", encoding="utf-8")

            write_env_updates(
                env_path,
                {
                    "TWINR_MEMORY_MAX_TURNS": "28",
                    "TWINR_MEMORY_KEEP_RECENT": "12",
                },
            )

            text = env_path.read_text(encoding="utf-8")

        self.assertIn("# existing", text)
        self.assertIn("TWINR_MEMORY_MAX_TURNS=28", text)
        self.assertIn("TWINR_MEMORY_KEEP_RECENT=12", text)

    def test_memory_capacity_level_uses_nearest_preset(self) -> None:
        level = memory_capacity_level(TwinrConfig(memory_max_turns=29, memory_keep_recent=12))

        self.assertEqual(level[0], 3)

    def test_current_voice_and_speed_helpers_return_normalized_values(self) -> None:
        config = TwinrConfig(
            openai_tts_voice="Marin",
            openai_realtime_voice="cedar",
            openai_tts_speed=0.9,
            openai_realtime_speed=1.1,
        )

        self.assertEqual(current_spoken_voice(config), "cedar")
        self.assertEqual(current_speech_speed(config), 1.0)
        self.assertIn("marin", supported_spoken_voices())

    def test_spoken_voice_language_note_uses_current_language_name(self) -> None:
        self.assertIn("German", spoken_voice_language_note("de"))
        self.assertIn("optimized for English", spoken_voice_language_note("de"))


if __name__ == "__main__":
    unittest.main()
