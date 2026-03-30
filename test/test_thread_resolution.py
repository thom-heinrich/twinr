from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.conversation.thread_resolution import (
    focus_recent_thread_conversation,
    maybe_rewrite_prompt_against_recent_thread,
    recent_thread_turns,
)


class ThreadResolutionTests(unittest.TestCase):
    def test_recent_thread_turns_ignore_system_messages(self) -> None:
        turns = recent_thread_turns(
            (
                ("system", "provider guidance"),
                ("system", "Twinr memory summary"),
                ("user", "Wie spaet ist es in New York?"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
            )
        )

        self.assertEqual(
            turns,
            (
                ("user", "Wie spaet ist es in New York?"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
            ),
        )

    def test_focus_recent_thread_conversation_keeps_recent_turns_and_carryover(self) -> None:
        conversation = focus_recent_thread_conversation(
            (
                ("system", "provider guidance"),
                ("system", "Twinr memory summary"),
                ("user", "Wie spaet ist es in New York?"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
            ),
            user_transcript="Ich meinte, wie spaet es ist.",
        )

        self.assertEqual(conversation[0], ("system", "provider guidance"))
        self.assertEqual(conversation[-2:], (
            ("user", "Wie spaet ist es in New York?"),
            ("assistant", "In New York ist es gerade 10:53 Uhr."),
        ))
        self.assertTrue(
            any(
                role == "system"
                and "Recent thread carryover for this turn." in content
                and "New York" in content
                for role, content in conversation
            )
        )

    def test_maybe_rewrite_prompt_against_recent_thread_returns_rewritten_prompt(self) -> None:
        with patch(
            "twinr.agent.base_agent.conversation.thread_resolution.request_structured_json_object",
            return_value={
                "resolution": "rewrite",
                "rewritten_user_text": "Wie spaet ist es in New York?",
            },
        ):
            resolution = maybe_rewrite_prompt_against_recent_thread(
                SimpleNamespace(
                    config=SimpleNamespace(default_model="gpt-test"),
                    _client=SimpleNamespace(),
                ),
                conversation=(
                    ("user", "Wie spaet ist es in New York?"),
                    ("assistant", "In New York ist es gerade 10:53 Uhr."),
                ),
                user_transcript="Ich meinte, wie spaet es ist.",
            )

        self.assertEqual(resolution.original_prompt, "Ich meinte, wie spaet es ist.")
        self.assertEqual(resolution.effective_prompt, "Wie spaet ist es in New York?")
        self.assertEqual(resolution.resolution, "rewrite")

    def test_maybe_rewrite_prompt_against_recent_thread_keeps_original_without_backend_shape(self) -> None:
        resolution = maybe_rewrite_prompt_against_recent_thread(
            object(),
            conversation=(
                ("user", "Wie spaet ist es in New York?"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
            ),
            user_transcript="Ich meinte, wie spaet es ist.",
        )

        self.assertEqual(resolution.effective_prompt, "Ich meinte, wie spaet es ist.")
        self.assertEqual(resolution.resolution, "keep")


if __name__ == "__main__":
    unittest.main()
