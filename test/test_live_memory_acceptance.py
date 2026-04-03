"""Targeted tests for the bounded live-memory acceptance context builder."""

from __future__ import annotations

import unittest

from twinr.agent.base_agent.config import TwinrConfig
from twinr.memory.longterm.core.models import (
    LongTermConflictOptionV1,
    LongTermConflictQueueItemV1,
    LongTermMemoryContext,
)
from twinr.memory.longterm.evaluation.live_memory_acceptance import (
    _AcceptanceCase,
    _acceptance_provider_conversation,
)


class _FakeAcceptanceService:
    """Provide just enough service surface for acceptance-context tests."""

    def __init__(self) -> None:
        self.fast_queries: list[str] = []
        self.tool_queries: list[tuple[str, bool]] = []
        self.conflict_queries: list[str] = []

    def build_fast_provider_context(self, query_text: str) -> LongTermMemoryContext:
        self.fast_queries.append(query_text)
        return LongTermMemoryContext(
            topic_context=(
                "twinr_fast_topic_context_v1\n"
                "- confirmed topic hint: Inzwischen magst du lieber Aprikosenmarmelade."
            )
        )

    def build_tool_provider_context(
        self,
        query_text: str,
        *,
        include_graph_fallback: bool = True,
    ) -> LongTermMemoryContext:
        self.tool_queries.append((query_text, include_graph_fallback))
        return LongTermMemoryContext(
            durable_context=(
                'twinr_long_term_durable_context_v1\n'
                '{"summary": "Inzwischen magst du lieber Aprikosenmarmelade.", '
                '"confirmed_by_user": true}'
            )
        )

    def select_conflict_queue(self, query_text: str) -> tuple[LongTermConflictQueueItemV1, ...]:
        self.conflict_queries.append(query_text)
        return (
            LongTermConflictQueueItemV1(
                slot_key="preference:breakfast:jam",
                question="Welche Marmelade stimmt gerade?",
                reason="Widerspruechliche Marmeladenpraeferenzen liegen vor.",
                candidate_memory_id="fact:jam_preference_new",
                options=(
                    LongTermConflictOptionV1(
                        memory_id="fact:jam_preference_old",
                        summary="Deine Lieblingsmarmelade ist Erdbeermarmelade.",
                        status="active",
                    ),
                    LongTermConflictOptionV1(
                        memory_id="fact:jam_preference_new",
                        summary="Inzwischen magst du lieber Aprikosenmarmelade.",
                        status="uncertain",
                    ),
                ),
            ),
        )


class LiveMemoryAcceptanceConversationTests(unittest.TestCase):
    """Verify the acceptance conversation stays bounded but memory-relevant."""

    def test_acceptance_provider_conversation_uses_fast_and_tool_context(self) -> None:
        service = _FakeAcceptanceService()
        config = TwinrConfig(project_root=".", personality_dir="personality", openai_realtime_language="de")
        case = _AcceptanceCase(
            case_id="resolved_meta_writer",
            phase="after_resolution",
            query_text="Welche Marmelade ist jetzt als bestaetigt gespeichert?",
            required_terms=("Aprikosenmarmelade",),
        )

        conversation = _acceptance_provider_conversation(
            service=service,  # type: ignore[arg-type]
            config=config,
            case=case,
        )

        system_messages = [content for role, content in conversation if role == "system"]
        self.assertTrue(any("Aprikosenmarmelade" in message for message in system_messages))
        self.assertTrue(any("confirmed_by_user" in message for message in system_messages))
        self.assertEqual(service.fast_queries, [case.query_text])
        self.assertEqual(service.tool_queries, [(case.query_text, False)])
        self.assertEqual(service.conflict_queries, [])

    def test_acceptance_provider_conversation_adds_conflict_queue_context_when_requested(self) -> None:
        service = _FakeAcceptanceService()
        config = TwinrConfig(project_root=".", personality_dir="personality", openai_realtime_language="de")
        case = _AcceptanceCase(
            case_id="conflict_before",
            phase="before_resolution",
            query_text="Welche Marmeladen stehen gerade im Widerspruch?",
            required_terms=("Erdbeermarmelade", "Aprikosenmarmelade"),
            include_conflict_queue_context=True,
        )

        conversation = _acceptance_provider_conversation(
            service=service,  # type: ignore[arg-type]
            config=config,
            case=case,
        )

        system_messages = "\n".join(content for role, content in conversation if role == "system")
        self.assertIn("twinr_acceptance_conflict_queue_v1", system_messages)
        self.assertIn("Erdbeermarmelade", system_messages)
        self.assertIn("Aprikosenmarmelade", system_messages)
        self.assertEqual(service.conflict_queries, [case.query_text])


if __name__ == "__main__":
    unittest.main()
