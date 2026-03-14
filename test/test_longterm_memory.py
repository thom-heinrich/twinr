from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.chonkydb import TwinrPersonalGraphStore
from twinr.memory.longterm import LongTermMemoryService
from twinr.memory.query_normalization import LongTermQueryProfile


class _StaticQueryRewriter:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def profile(self, query_text: str | None) -> LongTermQueryProfile:
        canonical = self._mapping.get(str(query_text or ""))
        return LongTermQueryProfile.from_text(query_text, canonical_english_text=canonical)


class LongTermMemoryServiceTests(unittest.TestCase):
    def test_background_worker_persists_episodic_turns_in_memory_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_write_queue_size=4,
            )
            service = LongTermMemoryService.from_config(config)
            result = service.enqueue_conversation_turn(
                transcript="Wie wird das Wetter heute?",
                response="Heute ist es sonnig und mild.",
            )
            drained = service.flush(timeout_s=2.0)
            entries = service.prompt_context_store.memory_store.load_entries()
            service.shutdown()

        self.assertIsNotNone(result)
        self.assertTrue(result.accepted)
        self.assertTrue(drained)
        self.assertEqual(entries[0].kind, "episodic_turn")
        self.assertIn('Conversation about "Wie wird das Wetter heute?"', entries[0].summary)
        self.assertIn('Twinr answered: "Heute ist es sonnig und mild."', entries[0].details or "")

    def test_provider_context_combines_recent_episodes_and_graph_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=2,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            service = LongTermMemoryService.from_config(config, graph_store=graph_store)
            service.enqueue_conversation_turn(
                transcript="Morgen wollte ich spazieren gehen.",
                response="Dann schaue ich morgen gern noch einmal auf das Wetter.",
            )
            service.flush(timeout_s=2.0)

            context = service.build_provider_context("Wie wird das Wetter heute?")
            service.shutdown()

        messages = context.system_messages()
        self.assertEqual(len(messages), 3)
        self.assertIn("Silent personalization background for this turn.", messages[0])
        self.assertIn("twinr_silent_personalization_context_v1", messages[0])
        self.assertIn("Structured long-term episodic memory for this turn.", messages[1])
        self.assertIn("Morgen wollte ich spazieren gehen.", messages[1])
        self.assertIn("twinr_graph_memory_context_v1", messages[2])
        self.assertIn("Melitta", messages[2])

    def test_subtext_context_surfaces_personalization_without_explicit_memory_language(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            graph_store.remember_plan(
                summary="Go for a walk in the park",
                when_text="today",
                details="The user wanted good weather for the walk.",
            )
            service = LongTermMemoryService.from_config(config, graph_store=graph_store)
            service.enqueue_conversation_turn(
                transcript="Tomorrow I want to go for a walk if the weather is nice.",
                response="I can keep the weather in mind for that walk.",
            )
            service.flush(timeout_s=2.0)

            context = service.build_provider_context("Where can I buy coffee today, and is the weather good?")
            service.shutdown()

        self.assertIsNotNone(context.subtext_context)
        subtext = context.subtext_context or ""
        self.assertIn("twinr_silent_personalization_context_v1", subtext)
        self.assertIn("Melitta", subtext)
        self.assertIn("Go for a walk in the park", subtext)
        self.assertIn("Use it as conversational subtext", subtext)
        self.assertNotIn("I remember", subtext)
        self.assertIn("Do not say earlier, before, last time, neulich", subtext)

    def test_query_rewrite_can_bridge_german_query_to_english_memory_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                long_term_memory_enabled=True,
                long_term_memory_recall_limit=3,
                long_term_memory_path=str(Path(temp_dir) / "state" / "chonkydb"),
                user_display_name="Erika",
            )
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(
                category="brand",
                value="Melitta",
                for_product="coffee",
            )
            service = LongTermMemoryService.from_config(config, graph_store=graph_store)
            service.query_rewriter = _StaticQueryRewriter(
                {"Wo kann ich heute Kaffee kaufen?": "Where can I buy coffee today?"}
            )

            context = service.build_provider_context("Wo kann ich heute Kaffee kaufen?")
            service.shutdown()

        self.assertIsNotNone(context.subtext_context)
        self.assertIn("Melitta", context.subtext_context or "")

    def test_explicit_memory_and_profile_updates_route_through_service(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
            )
            service = LongTermMemoryService.from_config(config)

            memory_entry = service.store_explicit_memory(
                kind="fact",
                summary="Preferred pharmacy is Linden Apotheke.",
                details="Use this as a stable reference.",
            )
            user_entry = service.update_user_profile(
                category="preferred_name",
                instruction="Call the user Erika.",
            )
            personality_entry = service.update_personality(
                category="response_style",
                instruction="Keep answers calm and short.",
            )

            memory_text = Path(config.memory_markdown_path).read_text(encoding="utf-8")
            user_text = (personality_dir / "USER.md").read_text(encoding="utf-8")
            personality_text = (personality_dir / "PERSONALITY.md").read_text(encoding="utf-8")

        self.assertEqual(memory_entry.kind, "fact")
        self.assertIn("Preferred pharmacy is Linden Apotheke.", memory_text)
        self.assertEqual(user_entry.key, "preferred_name")
        self.assertIn("preferred_name: Call the user Erika.", user_text)
        self.assertEqual(personality_entry.key, "response_style")
        self.assertIn("response_style: Keep answers calm and short.", personality_text)


if __name__ == "__main__":
    unittest.main()
