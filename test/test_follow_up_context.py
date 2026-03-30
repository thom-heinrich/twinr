from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.conversation.follow_up_context import (
    append_recent_thread_carryover_message,
    build_recent_thread_carryover_hint,
    build_follow_up_context_hint,
    follow_up_context_hint_trace_details,
    pending_conversation_follow_up_hint_trace_details,
    pending_conversation_follow_up_hint_scope,
    pending_conversation_follow_up_system_message,
    remember_pending_conversation_follow_up_hint,
)


class FollowUpContextTests(unittest.TestCase):
    def test_build_follow_up_context_hint_keeps_recent_anchor_turns(self) -> None:
        hint = build_follow_up_context_hint(
            conversation=(
                ("user", "wie spaet ist es in New York"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
                ("user", "mich"),
                ("assistant", "Meinst du etwas Bestimmtes?"),
            ),
            user_transcript="mich",
            assistant_response="Meinst du etwas Bestimmtes?",
        )

        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("New York", hint)
        self.assertIn("Meinst du etwas Bestimmtes?", hint)
        self.assertIn("Continue this thread", hint)

    def test_build_recent_thread_carryover_hint_keeps_recent_anchor_turns(self) -> None:
        hint = build_recent_thread_carryover_hint(
            conversation=(
                ("user", "wie spaet ist es in New York"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
                ("user", "mich"),
                ("assistant", "Meinst du etwas Bestimmtes?"),
            ),
            user_transcript="Ich meinte, wie spaet es ist.",
        )

        self.assertIsNotNone(hint)
        assert hint is not None
        self.assertIn("New York", hint)
        self.assertIn("Ich meinte, wie spaet es ist.", hint)
        self.assertIn("repair, clarification, or continuation", hint)

    def test_append_recent_thread_carryover_message_adds_system_guidance(self) -> None:
        conversation = append_recent_thread_carryover_message(
            (
                ("system", "memory summary"),
                ("user", "wie spaet ist es in New York"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
            ),
            user_transcript="Ich meinte, wie spaet es ist.",
        )

        self.assertTrue(
            any(
                role == "system"
                and "Recent thread carryover for this turn." in content
                and "New York" in content
                for role, content in conversation
            )
        )

    def test_pending_system_message_is_only_injected_while_scope_is_active(self) -> None:
        runtime = SimpleNamespace()
        remember_pending_conversation_follow_up_hint(
            runtime,
            summary="Recent turns: user=wie spaet ist es in New York; assistant=In New York ist es gerade 10:53 Uhr.",
        )

        self.assertIsNone(pending_conversation_follow_up_system_message(runtime))
        with pending_conversation_follow_up_hint_scope(runtime, active=True):
            message = pending_conversation_follow_up_system_message(runtime)

        self.assertIsNotNone(message)
        assert message is not None
        self.assertIn("Preserve explicit anchors", message)
        self.assertIn("New York", message)
        self.assertIsNone(pending_conversation_follow_up_system_message(runtime))

    def test_follow_up_context_hint_trace_details_redact_text_but_keep_hashes(self) -> None:
        details = follow_up_context_hint_trace_details(
            conversation=(
                ("user", "wie spaet ist es in New York"),
                ("assistant", "In New York ist es gerade 10:53 Uhr."),
            ),
            user_transcript="Ich meinte, wie spaet es ist.",
            assistant_response="Ich pruefe kurz die aktuelle Zeit fuer dich.",
            summary="Immediate open thread from the latest exchange.",
        )

        self.assertEqual(details["recent_turn_count"], 4)
        self.assertEqual(details["recent_roles"], ["user", "assistant", "user", "assistant"])
        self.assertTrue(details["summary"]["present"])
        self.assertEqual(details["summary"]["chars"], 47)
        self.assertNotIn("New York", str(details))
        self.assertEqual(len(details["user_transcript"]["sha256_12"]), 12)

    def test_pending_follow_up_hint_trace_details_report_activation_without_raw_text(self) -> None:
        runtime = SimpleNamespace()
        remember_pending_conversation_follow_up_hint(
            runtime,
            summary="Recent turns: user=wie spaet ist es in New York; assistant=In New York ist es gerade 10:53 Uhr.",
        )

        inactive = pending_conversation_follow_up_hint_trace_details(runtime)
        self.assertFalse(inactive["active"])
        self.assertTrue(inactive["summary"]["present"])
        with pending_conversation_follow_up_hint_scope(runtime, active=True):
            active = pending_conversation_follow_up_hint_trace_details(runtime)

        self.assertTrue(active["active"])
        self.assertEqual(len(active["summary"]["sha256_12"]), 12)
        self.assertNotIn("New York", str(active))


if __name__ == "__main__":
    unittest.main()
