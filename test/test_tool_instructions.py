from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools.instructions import (
    build_compact_tool_agent_instructions,
    build_tool_agent_instructions,
)
from twinr.config import TwinrConfig


class ToolInstructionTests(unittest.TestCase):
    def test_default_tool_instructions_prevent_repeated_search_calls_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("do not call search_live_info again", instructions)
        self.assertIn("exact requested detail could not be verified", instructions)

    def test_compact_tool_instructions_prevent_repeated_search_calls_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_compact_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("do not call it again", instructions)
        self.assertIn("exact requested detail", instructions)

    def test_tool_instructions_require_exact_retrieval_tools_before_answering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("exact saved contact details", instructions)
        self.assertIn("lookup or list tool first", instructions)
        self.assertIn("switch from one supported trigger type to another", instructions)


if __name__ == "__main__":
    unittest.main()
