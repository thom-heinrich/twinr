from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.evaluation.live_midterm_acceptance import _build_isolated_config


class LiveLongTermAcceptanceConfigTests(unittest.TestCase):
    def test_build_isolated_config_reroots_runtime_owned_state_files(self) -> None:
        with tempfile.TemporaryDirectory() as base_dir, tempfile.TemporaryDirectory() as runtime_dir:
            base_root = Path(base_dir)
            runtime_root = Path(runtime_dir)
            personality_dir = base_root / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "system.txt").write_text("system", encoding="utf-8")
            base_config = TwinrConfig(
                project_root=str(base_root),
                personality_dir="personality",
                memory_markdown_path="state/MEMORY.md",
                runtime_state_path="state/runtime-state.json",
                reminder_store_path="state/reminders.json",
                automation_store_path="state/automations.json",
                voice_profile_store_path="state/voice_profile.json",
                adaptive_timing_store_path="state/adaptive_timing.json",
                long_term_memory_path="state/chonkydb",
            )

            isolated = _build_isolated_config(
                base_config=base_config,
                base_project_root=base_root,
                runtime_root=runtime_root,
                remote_namespace="test-namespace",
                background_store_turns=False,
            )

            state_dir = runtime_root / "state"
            self.assertEqual(Path(isolated.project_root), runtime_root)
            self.assertEqual(Path(isolated.memory_markdown_path), state_dir / "MEMORY.md")
            self.assertEqual(Path(isolated.runtime_state_path), runtime_root / "runtime-state.json")
            self.assertEqual(Path(isolated.reminder_store_path), state_dir / "reminders.json")
            self.assertEqual(Path(isolated.automation_store_path), state_dir / "automations.json")
            self.assertEqual(Path(isolated.voice_profile_store_path), state_dir / "voice_profile.json")
            self.assertEqual(Path(isolated.adaptive_timing_store_path), state_dir / "adaptive_timing.json")
            self.assertEqual(Path(isolated.long_term_memory_path), state_dir / "chonkydb")
            self.assertTrue((runtime_root / "personality" / "system.txt").is_file())


if __name__ == "__main__":
    unittest.main()
