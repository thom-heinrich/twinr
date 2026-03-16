from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools.prompting.instructions import (
    build_compact_tool_agent_instructions,
    build_first_word_instructions,
    build_supervisor_decision_instructions,
    build_supervisor_tool_agent_instructions,
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
        self.assertIn("propose_skill_learning", instructions)
        self.assertIn("answer_skill_question", instructions)

    def test_supervisor_instructions_do_not_embed_full_business_tool_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_supervisor_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertNotIn("remember_memory tool", instructions)
        self.assertNotIn("update_user_profile tool", instructions)
        self.assertNotIn("remember_contact tool", instructions)
        self.assertIn("call handoff_specialist_worker", instructions)

    def test_supervisor_instructions_forbid_claiming_persistence_without_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_supervisor_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("Never claim that something was saved", instructions)
        self.assertIn("settings actions directly", instructions)
        self.assertIn("handoff_specialist_worker.spoken_ack", instructions)
        self.assertIn("Do not wait for the specialist result", instructions)

    def test_supervisor_decision_instructions_use_structured_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_supervisor_decision_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("three structured actions", instructions)
        self.assertIn("Choose handoff", instructions)
        self.assertIn("set spoken_ack", instructions)
        self.assertIn("one or two short sentences", instructions)
        self.assertIn("without sounding canned", instructions)
        self.assertIn("must not imply the task is already finished", instructions)
        self.assertIn("put the full user-facing answer into spoken_reply", instructions)
        self.assertIn("Do not wait for the specialist result", instructions)

    def test_first_word_instructions_encourage_warm_conversational_replies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_first_word_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("one or two short sentences", instructions)
        self.assertIn("short warm follow-up question", instructions)
        self.assertIn("relevant remembered user detail", instructions)


if __name__ == "__main__":
    unittest.main()
