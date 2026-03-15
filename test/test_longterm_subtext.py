from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.context_store import PromptContextStore
from twinr.memory.longterm.conflicts import LongTermConflictResolver
from twinr.memory.longterm.models import LongTermConsolidationResultV1, LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.midterm_store import LongTermMidtermStore
from twinr.memory.longterm.retriever import LongTermRetriever
from twinr.memory.longterm.store import LongTermStructuredStore
from twinr.memory.longterm.subtext import LongTermSubtextBuilder, LongTermSubtextCompiler
from twinr.memory.query_normalization import LongTermQueryProfile


class _FakeResponses:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def create(self, **_request):
        self.calls += 1
        return types.SimpleNamespace(output_parsed=self.payload)


class _FakeBackend:
    def __init__(self, payload: dict[str, object]) -> None:
        self.config = types.SimpleNamespace(default_model="gpt-test")
        self._responses = _FakeResponses(payload)
        self._client = types.SimpleNamespace(responses=self._responses)

    def _build_response_request(self, *_args, **_kwargs):
        return {}

    def _extract_output_text(self, _response) -> str:
        return ""


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_recall_limit=3,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
        user_display_name="Erika",
    )


class LongTermSubtextBuilderTests(unittest.TestCase):
    def test_compiled_program_is_used_when_compiler_backend_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(category="brand", value="Melitta", for_product="coffee")
            graph_store.remember_plan(
                summary="Go for a walk in the park",
                when_text="today",
                details="The user wanted good weather for the walk.",
            )
            compiler = LongTermSubtextCompiler(
                config=config,
                backend=_FakeBackend(
                    {
                        "use_personalization": True,
                        "conversation_goal": "Quietly favor a walk-oriented outdoor suggestion.",
                        "helpful_biases": ["Treat the walk as the most relevant outdoor option."],
                        "suggested_directions": ["Prefer a weather-aware walking suggestion over generic brainstorming."],
                        "follow_up_angles": ["Ask one short weather or time-availability follow-up if needed."],
                        "known_people": [],
                        "avoidances": ["Do not announce that this came from memory."],
                    }
                ),
                _cache={},
            )
            builder = LongTermSubtextBuilder(config=config, graph_store=graph_store, compiler=compiler)
            prompt_store = PromptContextStore.from_config(config)
            prompt_store.memory_store.remember(
                kind="episodic_turn",
                summary='Conversation about "Tomorrow I want to go for a walk if the weather is nice."',
                details='User said: "Tomorrow I want to go for a walk if the weather is nice." Twinr answered: "I can keep the weather in mind for that walk."',
            )
            episodic_entries = prompt_store.memory_store.load_entries()

            text = builder.build(
                query_text="Ich ueberlege, was heute ein guter Plan fuer draussen waere.",
                retrieval_query_text="I am thinking about a good outdoor plan for today.",
                episodic_entries=episodic_entries,
            )

        self.assertIsNotNone(text)
        rendered = text or ""
        self.assertIn("twinr_silent_personalization_program_v3", rendered)
        self.assertIn("Quietly favor a walk-oriented outdoor suggestion", rendered)
        self.assertIn("Relevant helpful biases:", rendered)
        self.assertNotIn("twinr_silent_personalization_context_v1", rendered)

    def test_compiler_failure_falls_back_to_static_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_preference(category="brand", value="Melitta", for_product="coffee")
            builder = LongTermSubtextBuilder(
                config=config,
                graph_store=graph_store,
                compiler=LongTermSubtextCompiler(config=config, backend=None, _cache={}),
            )

            text = builder.build(
                query_text="Wo kann ich heute Kaffee kaufen?",
                retrieval_query_text="Where can I buy coffee today?",
                episodic_entries=(),
            )

        self.assertIsNotNone(text)
        rendered = text or ""
        self.assertIn("twinr_silent_personalization_context_v1", rendered)
        self.assertIn("Melitta", rendered)


class LongTermRetrieverCompiledSubtextTests(unittest.TestCase):
    def test_retriever_emits_compiled_subtext_program_when_builder_compiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_plan(
                summary="Go for a walk in the park",
                when_text="today",
                details="The user wanted good weather for the walk.",
            )
            compiler = LongTermSubtextCompiler(
                config=config,
                backend=_FakeBackend(
                    {
                        "use_personalization": True,
                        "conversation_goal": "Quietly make the answer action-oriented around the outdoor walk.",
                        "helpful_biases": ["Prefer the existing walk plan when discussing outdoor options."],
                        "suggested_directions": ["Suggest checking weather and time for the walk."],
                        "follow_up_angles": ["Ask whether the user wants a short or longer walk."],
                        "known_people": [],
                        "avoidances": ["Do not narrate memory provenance."],
                    }
                ),
                _cache={},
            )
            retriever = LongTermRetriever(
                config=config,
                prompt_context_store=PromptContextStore.from_config(config),
                graph_store=graph_store,
                object_store=LongTermStructuredStore.from_config(config),
                midterm_store=LongTermMidtermStore.from_config(config),
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=LongTermSubtextBuilder(
                    config=config,
                    graph_store=graph_store,
                    compiler=compiler,
                ),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Ich ueberlege, was heute ein guter Plan fuer draussen waere.",
                    canonical_english_text="I am thinking about a good outdoor plan for today.",
                ),
                original_query_text="Ich ueberlege, was heute ein guter Plan fuer draussen waere.",
            )

        self.assertIsNotNone(context.subtext_context)
        self.assertIn("twinr_silent_personalization_program_v3", context.subtext_context or "")
        self.assertIn("outdoor walk", context.subtext_context or "")

    def test_compiler_sanitizes_structured_program_strings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            compiler = LongTermSubtextCompiler(
                config=config,
                backend=_FakeBackend(
                    {
                        "use_personalization": True,
                        "conversation_goal": " Help decide whether to call Corinna today.\u200f ",
                        "helpful_biases": [" Keep guidance practical.\u200f "],
                        "suggested_directions": [" Ask about appointment timing.\u200f "],
                        "follow_up_angles": [" Is there a therapy question for today?\u200f "],
                        "known_people": [
                            {
                                "person": " Corinna Maier\u200f ",
                                "role_or_relation": " physiotherapist\u200f ",
                                "latent_relevance": " Therapy scheduling may matter today.\u200f ",
                                "practical_topics": [" appointment timing\u200f ", " exercises\u200f "],
                            }
                        ],
                        "avoidances": [" Do not announce memory.\u200f "],
                    }
                ),
                _cache={},
            )

            payload = compiler.compile(
                query_text="Soll ich Corinna heute noch anrufen?",
                retrieval_query_text="Should I call Corinna today?",
                graph_payload={
                    "social_context": [
                        {
                            "person": "Corinna Maier",
                            "role": "physiotherapist",
                            "guidance": "Use the role as hidden practical context.",
                        }
                    ]
                },
                recent_threads=(),
            )

        self.assertIsNotNone(payload)
        self.assertEqual(payload["conversation_goal"], "Help decide whether to call Corinna today.")
        known_people = payload["known_people"]
        self.assertEqual(known_people[0]["person"], "Corinna Maier")
        self.assertEqual(known_people[0]["role_or_relation"], "physiotherapist")
        self.assertEqual(known_people[0]["practical_topics"], ["appointment timing", "exercises"])

    def test_rendered_directives_keep_known_person_frame_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            graph_store = TwinrPersonalGraphStore.from_config(config)
            graph_store.remember_contact(
                given_name="Corinna",
                family_name="Maier",
                role="physiotherapist",
            )
            compiler = LongTermSubtextCompiler(
                config=config,
                backend=_FakeBackend(
                    {
                        "use_personalization": True,
                        "conversation_goal": "Help decide whether to call Corinna today.",
                        "helpful_biases": ["Keep the guidance practical."],
                        "suggested_directions": ["Frame the answer around physiotherapy needs."],
                        "follow_up_angles": ["Ask about pain or appointment changes."],
                        "known_people": [
                            {
                                "person": "Corinna Maier",
                                "role_or_relation": "physiotherapist",
                                "latent_relevance": "Calling may help with pain changes or scheduling.",
                                "practical_topics": ["pain changes", "appointment scheduling"],
                            }
                        ],
                        "avoidances": ["Do not mention hidden memory."],
                    }
                ),
                _cache={},
            )
            builder = LongTermSubtextBuilder(
                config=config,
                graph_store=graph_store,
                compiler=compiler,
            )

            text = builder.build(
                query_text="Soll ich Corinna heute noch anrufen?",
                retrieval_query_text="Should I call Corinna today?",
                episodic_entries=(),
            )

        rendered = text or ""
        self.assertIn("If Corinna Maier is directly relevant, treat them as physiotherapist.", rendered)
        self.assertIn("Useful practical angles: pain changes; appointment scheduling.", rendered)
        self.assertIn("If a concrete recommendation or decision is being made", rendered)

    def test_retriever_uses_recent_episode_for_subtext_even_without_literal_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            prompt_store = PromptContextStore.from_config(config)
            object_store = LongTermStructuredStore.from_config(config)
            episode = LongTermMemoryObjectV1(
                memory_id="episode:knee",
                kind="episode",
                summary='Conversation about "My knee hurts a bit today."',
                details='User said: "My knee hurts a bit today." Twinr answered: "Then it may be wise to take it easy today."',
                source=LongTermSourceRefV1(
                    source_type="conversation_turn",
                    event_ids=("turn:knee",),
                    speaker="user",
                    modality="voice",
                ),
                status="active",
                confidence=1.0,
                attributes={
                    "raw_transcript": "My knee hurts a bit today.",
                    "raw_response": "Then it may be wise to take it easy today.",
                },
            )
            object_store.apply_consolidation(
                LongTermConsolidationResultV1(
                    turn_id="turn:knee",
                    occurred_at=episode.created_at,
                    episodic_objects=(episode,),
                    durable_objects=(),
                    deferred_objects=(),
                    conflicts=(),
                    graph_edges=(),
                )
            )
            compiler = LongTermSubtextCompiler(
                config=config,
                backend=_FakeBackend(
                    {
                        "use_personalization": True,
                        "conversation_goal": "Quietly factor recent knee discomfort into the walking advice.",
                        "helpful_biases": ["Favor a gentler short walk or rest option."],
                        "suggested_directions": ["Treat the decision as comfort and strain management."],
                        "follow_up_angles": ["Ask whether the knee still hurts today."],
                        "known_people": [],
                        "avoidances": ["Do not announce prior memory."],
                    }
                ),
                _cache={},
            )
            retriever = LongTermRetriever(
                config=config,
                prompt_context_store=prompt_store,
                graph_store=TwinrPersonalGraphStore.from_config(config),
                object_store=object_store,
                midterm_store=LongTermMidtermStore.from_config(config),
                conflict_resolver=LongTermConflictResolver(),
                subtext_builder=LongTermSubtextBuilder(
                    config=config,
                    graph_store=TwinrPersonalGraphStore.from_config(config),
                    compiler=compiler,
                ),
            )

            context = retriever.build_context(
                query=LongTermQueryProfile.from_text(
                    "Soll ich heute spazieren gehen oder lieber zuhause bleiben?",
                    canonical_english_text="Should I go for a walk today or stay home?",
                ),
                original_query_text="Soll ich heute spazieren gehen oder lieber zuhause bleiben?",
            )

        self.assertIsNotNone(context.subtext_context)
        self.assertIn("knee discomfort", context.subtext_context or "")
        self.assertIn("Favor a gentler short walk or rest option.", context.subtext_context or "")


if __name__ == "__main__":
    unittest.main()
