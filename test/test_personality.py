from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.personality import load_personality_instructions, merge_instructions


class PersonalityTests(unittest.TestCase):
    def test_load_personality_instructions_orders_sections_for_stable_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            personality_dir = Path(tmpdir) / "personality"
            personality_dir.mkdir()
            (personality_dir / "SYSTEM.md").write_text("System context", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Style context", encoding="utf-8")
            (personality_dir / "USER.md").write_text("User profile", encoding="utf-8")

            instructions = load_personality_instructions(
                TwinrConfig(
                    project_root=tmpdir,
                    personality_dir="personality",
                )
            )

        self.assertIsNotNone(instructions)
        self.assertLess(instructions.index("SYSTEM:\nSystem context"), instructions.index("PERSONALITY:\nStyle context"))
        self.assertLess(instructions.index("PERSONALITY:\nStyle context"), instructions.index("USER:\nUser profile"))

    def test_merge_instructions_skips_empty_parts(self) -> None:
        merged = merge_instructions("Base", None, " ", "Task")
        self.assertEqual(merged, "Base\n\nTask")


if __name__ == "__main__":
    unittest.main()
