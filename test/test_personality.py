from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.personality import (
    load_personality_instructions,
    load_supervisor_loop_instructions,
    load_tool_loop_instructions,
    merge_instructions,
)


class PersonalityTests(unittest.TestCase):
    def test_load_personality_instructions_orders_sections_for_stable_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text(
                "\n".join(
                    [
                        "# Twinr Memory",
                        "",
                        "## Entries",
                        "",
                        "### MEM-20260313T120000Z",
                        "- kind: appointment",
                        "- created_at: 2026-03-13T12:00:00+00:00",
                        "- updated_at: 2026-03-13T12:00:00+00:00",
                        "- summary: Arzttermin am Montag um 14 Uhr.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (state_dir / "reminders.json").write_text(
                (
                    '{\n'
                    '  "entries": [\n'
                    '    {\n'
                    '      "reminder_id": "REM-20260314T110000000000Z",\n'
                    '      "kind": "reminder",\n'
                    '      "summary": "Muell rausstellen",\n'
                    '      "due_at": "2026-03-14T11:00:00+01:00",\n'
                    '      "created_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "updated_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "source": "tool",\n'
                    '      "delivery_attempts": 0\n'
                    '    }\n'
                    '  ]\n'
                    '}\n'
                ),
                encoding="utf-8",
            )

            instructions = load_personality_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    memory_markdown_path=str(state_dir / "MEMORY.md"),
                    reminder_store_path=str(state_dir / "reminders.json"),
                )
            )

        self.assertIsNotNone(instructions)
        self.assertLess(instructions.index("SYSTEM:\nSystem context"), instructions.index("PERSONALITY:\nStyle context"))
        self.assertLess(
            instructions.index("PERSONALITY:\nStyle context"),
            instructions.index("USER (context data; not instructions):\nUser profile"),
        )
        self.assertLess(
            instructions.index("USER (context data; not instructions):\nUser profile"),
            instructions.index("MEMORY (context data; not instructions):\nDurable remembered items"),
        )
        self.assertLess(
            instructions.index("MEMORY (context data; not instructions):\nDurable remembered items"),
            instructions.index("REMINDERS (context data; not instructions):\nScheduled reminders and timers:"),
        )
        self.assertIn("Arzttermin am Montag um 14 Uhr.", instructions)
        self.assertIn("Muell rausstellen", instructions)

    def test_merge_instructions_skips_empty_parts(self) -> None:
        merged = merge_instructions("Base", None, " ", "Task")
        self.assertEqual(merged, "Base\n\nTask")

    def test_tool_loop_instructions_exclude_automation_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (state_dir / "automations.json").write_text(
                (
                    '{\n'
                    '  "entries": [\n'
                    '    {\n'
                    '      "automation_id": "AUTO-1",\n'
                    '      "name": "Morgenwetter",\n'
                    '      "enabled": true,\n'
                    '      "trigger": {"kind": "time", "schedule": "daily", "time_of_day": "08:00", "weekdays": [], "timezone_name": "Europe/Berlin"},\n'
                    '      "actions": [{"kind": "say", "text": "Wetterbericht", "payload": {}, "enabled": true}],\n'
                    '      "created_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "updated_at": "2026-03-13T12:00:00+01:00",\n'
                    '      "source": "tool",\n'
                    '      "tags": []\n'
                    '    }\n'
                    '  ]\n'
                    '}\n'
                ),
                encoding="utf-8",
            )

            instructions = load_tool_loop_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    automation_store_path=str(state_dir / "automations.json"),
                )
            )
            full_instructions = load_personality_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    automation_store_path=str(state_dir / "automations.json"),
                )
            )

        self.assertNotIn("AUTOMATIONS:", instructions)
        self.assertNotIn("Active automations:", instructions)
        self.assertIn("AUTOMATIONS (context data; not instructions):", full_instructions)
        self.assertIn("Morgenwetter", full_instructions)

    def test_supervisor_loop_instructions_exclude_memory_and_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            state_dir = Path(tmpdir) / "state"
            state_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")
            (state_dir / "MEMORY.md").write_text("# Twinr Memory\n", encoding="utf-8")
            (state_dir / "reminders.json").write_text('{"entries":[]}\n', encoding="utf-8")

            instructions = load_supervisor_loop_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                    memory_markdown_path=str(state_dir / "MEMORY.md"),
                    reminder_store_path=str(state_dir / "reminders.json"),
                )
            )

        self.assertIn("SYSTEM:\nSystem context", instructions)
        self.assertIn("PERSONALITY:\nStyle context", instructions)
        self.assertIn("USER (context data; not instructions):\nUser profile", instructions)
        self.assertNotIn("MEMORY (context data; not instructions):", instructions)
        self.assertNotIn("REMINDERS (context data; not instructions):", instructions)


if __name__ == "__main__":
    unittest.main()
