from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools.handlers.user_discovery import handle_manage_user_discovery
from twinr.agent.tools.runtime.registry import realtime_tool_names
from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.display.reserve_bus_feedback import DisplayReserveBusFeedbackStore
from twinr.memory.chonkydb.personal_graph import TwinrPersonalGraphStore
from twinr.memory.longterm.core.models import LongTermMemoryObjectV1, LongTermSourceRefV1
from twinr.memory.longterm.storage.store import LongTermStructuredStore
from twinr.memory.user_discovery import (
    UserDiscoveryCommitCallbacks,
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryService,
    UserDiscoveryState,
    UserDiscoveryStoredFact,
    UserDiscoveryTopicState,
)
from twinr.memory.user_discovery_authoritative_profile import UserDiscoveryAuthoritativeProfileReader
from twinr.memory.user_discovery_impl.store import UserDiscoveryStateStore
from twinr.proactive.runtime.display_reserve_user_discovery import load_display_reserve_user_discovery_candidates


def _active_preference_object(
    *,
    memory_id: str,
    summary: str,
    attributes: dict[str, object],
) -> LongTermMemoryObjectV1:
    return LongTermMemoryObjectV1(
        memory_id=memory_id,
        kind="fact",
        summary=summary,
        source=LongTermSourceRefV1(source_type="user_transcript", event_ids=("turn:test",)),
        status="active",
        confirmed_by_user=True,
        attributes=attributes,
    )


class UserDiscoveryTests(unittest.TestCase):
    def test_authoritative_profile_reader_uses_targeted_object_queries_before_graph_blob_reads(self) -> None:
        basics_object = _active_preference_object(
            memory_id="fact:user:main:prefers_name:thom",
            summary="The user wants to be called Thom.",
            attributes={
                "subject_ref": "user:main",
                "predicate": "prefers_name",
                "preference_type": "name",
                "preference_value": "Thom",
            },
        )
        style_object = _active_preference_object(
            memory_id="pref:initiative:gentle",
            summary="The user likes gentle initiative in follow-ups.",
            attributes={
                "subject_ref": "user:main",
                "predicate": "user_prefers_small_follow_up_when_helpful",
                "preference_type": "initiative",
                "preference_value": "gently_proactive",
            },
        )

        class QueryOnlyObjectStore:
            def __init__(self) -> None:
                self.queries: list[tuple[str | None, int]] = []

            def select_fast_topic_objects(
                self,
                *,
                query_text: str | None,
                limit: int = 4,
            ) -> tuple[LongTermMemoryObjectV1, ...]:
                self.queries.append((query_text, limit))
                normalized_query = str(query_text or "")
                if "preference_type name" in normalized_query and "prefers_name" in normalized_query:
                    return (basics_object,)
                if "preference_type initiative" in normalized_query:
                    return (style_object,)
                return ()

            def load_objects(self) -> tuple[LongTermMemoryObjectV1, ...]:
                raise AssertionError("Full structured object snapshot loads are forbidden here.")

        class NoBlobGraphStore:
            def load_document(self) -> None:
                raise AssertionError("Graph blob loads should stay lazy when targeted object queries already cover the topics.")

        object_store = QueryOnlyObjectStore()
        reader = UserDiscoveryAuthoritativeProfileReader(
            graph_store=NoBlobGraphStore(),
            object_store=object_store,
        )

        coverage = reader.load()

        self.assertTrue(coverage.covers("basics"))
        self.assertTrue(coverage.covers("companion_style"))
        self.assertEqual(len(object_store.queries), 2)
        self.assertTrue(any("preference_type name" in str(query) for query, _limit in object_store.queries))
        self.assertTrue(any("preference_type initiative" in str(query) for query, _limit in object_store.queries))

    def test_runtime_user_discovery_commits_high_value_user_profile_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            runtime = TwinrRuntime(config=config)
            try:
                start = runtime.manage_user_discovery(action="start_or_resume")
                answer = runtime.manage_user_discovery(
                    action="answer",
                    learned_facts=(
                        UserDiscoveryFact(
                            storage="user_profile",
                            text="User prefers to be called Thom.",
                        ),
                    ),
                    topic_complete=True,
                )
            finally:
                runtime.shutdown(timeout_s=0.1)

            user_text = (personality_dir / "USER.md").read_text(encoding="utf-8")
            discovery_state = json.loads((Path(temp_dir) / "state" / "user_discovery.json").read_text(encoding="utf-8"))

        self.assertEqual(start.response_mode, "ask_question")
        self.assertEqual(start.topic_id, "basics")
        self.assertEqual(answer.response_mode, "ask_question")
        self.assertEqual(answer.topic_id, "companion_style")
        self.assertIn("user_discovery_basics: User prefers to be called Thom.", user_text)
        self.assertEqual(discovery_state["active_topic_id"], "companion_style")
        self.assertEqual(discovery_state["phase"], "initial_setup")

    def test_state_store_save_writes_cross_service_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "state" / "user_discovery.json"
            store = UserDiscoveryStateStore(path=path)

            store.save(UserDiscoveryState.empty())
            store.save(UserDiscoveryState.empty())

            mode = path.stat().st_mode & 0o777

        self.assertEqual(mode, 0o644)

    def test_handler_can_store_personality_learning_and_emit_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            runtime = TwinrRuntime(config=config)
            lines: list[str] = []
            events: list[tuple[str, str, dict[str, object]]] = []
            owner = SimpleNamespace(
                runtime=runtime,
                emit=lines.append,
                _record_event=lambda event_name, message, **metadata: events.append((event_name, message, metadata)),
            )
            try:
                start = handle_manage_user_discovery(
                    owner,
                    {"action": "start_or_resume", "topic_id": "companion_style"},
                )
                answer = handle_manage_user_discovery(
                    owner,
                    {
                        "action": "answer",
                        "learned_facts": [
                            {
                                "storage": "personality",
                                "text": "Use informal German Du with the user.",
                            }
                        ],
                        "topic_complete": True,
                        "confirmed": True,
                    },
                )
            finally:
                runtime.shutdown(timeout_s=0.1)

            personality_text = (personality_dir / "PERSONALITY.md").read_text(encoding="utf-8")

        self.assertEqual(start["response_mode"], "ask_question")
        self.assertEqual(start["topic_id"], "companion_style")
        self.assertEqual(answer["response_mode"], "ask_question")
        self.assertIn("user_discovery_companion_style: Use informal German Du with the user.", personality_text)
        self.assertIn("user_discovery_tool_call=True", lines)
        self.assertIn("user_discovery_action=answer", lines)
        self.assertTrue(any(event_name == "user_discovery_updated" for event_name, _message, _metadata in events))

    def test_service_gates_health_topic_on_explicit_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            service = UserDiscoveryService.from_config(config)
            start = service.manage(action="start_or_resume", topic_id="health")
            declined = service.manage(action="answer", topic_id="health", permission_granted=False)

        self.assertEqual(start.response_mode, "ask_permission")
        self.assertTrue(start.sensitive_permission_required)
        self.assertEqual(declined.response_mode, "ask_question")
        self.assertEqual(declined.topic_id, "basics")

    def test_review_replace_delete_flow_updates_managed_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            runtime = TwinrRuntime(config=config)
            try:
                runtime.manage_user_discovery(action="start_or_resume", topic_id="basics")
                runtime.manage_user_discovery(
                    action="answer",
                    learned_facts=(UserDiscoveryFact(storage="user_profile", text="User prefers to be called Thom."),),
                )
                review = runtime.manage_user_discovery(action="review_profile")
                fact_id = review.review_items[0].fact_id
                runtime.manage_user_discovery(
                    action="replace_fact",
                    fact_id=fact_id,
                    memory_routes=(
                        UserDiscoveryMemoryRoute(
                            route_kind="user_profile",
                            text="User prefers to be called Tom.",
                        ),
                    ),
                )
                replaced_review = runtime.manage_user_discovery(action="review_profile")
                replaced_fact_id = replaced_review.review_items[0].fact_id
                runtime.manage_user_discovery(action="delete_fact", fact_id=replaced_fact_id)
                final_review = runtime.manage_user_discovery(action="review_profile")
            finally:
                runtime.shutdown(timeout_s=0.1)

            user_text = (personality_dir / "USER.md").read_text(encoding="utf-8")

        self.assertEqual(review.response_mode, "review_profile")
        self.assertEqual(replaced_review.response_mode, "review_profile")
        self.assertFalse(final_review.review_items)
        self.assertNotIn("Thom", user_text)
        self.assertNotIn("Tom", user_text)

    def test_runtime_discovery_routes_structured_memory_to_graph_and_durable_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            runtime = TwinrRuntime(config=config)
            try:
                runtime.manage_user_discovery(action="start_or_resume", topic_id="social")
                runtime.manage_user_discovery(
                    action="answer",
                    memory_routes=(
                        UserDiscoveryMemoryRoute(route_kind="contact", given_name="Anna", relation="daughter"),
                        UserDiscoveryMemoryRoute(route_kind="preference", category="drink", value="Melitta", sentiment="prefer"),
                        UserDiscoveryMemoryRoute(route_kind="plan", summary="Call Anna", when_text="tomorrow"),
                        UserDiscoveryMemoryRoute(
                            route_kind="durable_memory",
                            kind="boundary",
                            summary="Do not call after 9 pm.",
                        ),
                    ),
                )
                lookup = runtime.lookup_contact(name="Anna")
                graph_context = runtime.graph_memory.build_prompt_context("Anna Melitta tomorrow")
            finally:
                runtime.shutdown(timeout_s=0.1)

            memory_text = (Path(temp_dir) / "state" / "MEMORY.md").read_text(encoding="utf-8")

        self.assertEqual(lookup.status, "found")
        self.assertIsNotNone(graph_context)
        self.assertIn("Anna", graph_context or "")
        self.assertIn("Melitta", graph_context or "")
        self.assertIn("Call Anna", graph_context or "")
        self.assertIn("Do not call after 9 pm.", memory_text)

    def test_lifelong_planner_prefers_topics_with_positive_discovery_engagement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            service = UserDiscoveryService.from_config(config)
            history = DisplayAmbientImpulseHistoryStore.from_config(config)
            now = datetime(2026, 3, 23, 18, 0, tzinfo=timezone.utc)
            exposure = history.append_exposure(
                source="user_discovery",
                topic_key="user_discovery:lifelong_learning:pets",
                title="Haustiere",
                headline="Haustiere",
                body="Ein paar Fragen zu Haustieren.",
                action="ask_one",
                attention_state="growing",
                shown_at=now - timedelta(days=1),
                expires_at=now - timedelta(days=1) + timedelta(minutes=10),
                metadata={"candidate_family": "user_discovery", "topic_id": "pets"},
            )
            history.resolve_feedback(
                exposure_id=exposure.exposure_id,
                response_status="engaged",
                response_sentiment="positive",
                response_at=now - timedelta(days=1) + timedelta(minutes=1),
                response_mode="voice_immediate_pickup",
                response_latency_seconds=6.0,
                response_turn_id="turn:1",
                response_target="Haustiere",
                response_summary="Immediate pickup.",
            )
            service.store.save(
                UserDiscoveryState(
                    phase="lifelong_learning",
                    session_state="idle",
                    setup_completed_at=(now - timedelta(days=2)).isoformat(),
                    topics=tuple(UserDiscoveryTopicState(topic_id=topic_id, completed_once=True) for topic_id in (
                        "basics",
                        "companion_style",
                        "social",
                        "interests",
                        "hobbies",
                        "routines",
                        "pets",
                        "no_goes",
                        "health",
                    )),
                )
            )

            invite = service.build_invitation(now=now)

        self.assertIsNotNone(invite)
        assert invite is not None
        self.assertEqual(invite.topic_id, "pets")

    def test_display_reserve_candidates_offer_review_when_corrections_accumulate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            service = UserDiscoveryService.from_config(config)
            now = datetime(2026, 3, 23, 18, 0, tzinfo=timezone.utc)
            service.store.save(
                UserDiscoveryState(
                    phase="lifelong_learning",
                    session_state="idle",
                    setup_completed_at=(now - timedelta(days=20)).isoformat(),
                    last_reviewed_at=(now - timedelta(days=3)).isoformat(),
                    topics=(
                        UserDiscoveryTopicState(
                            topic_id="basics",
                            correction_count=1,
                            last_answer_at=(now - timedelta(days=1)).isoformat(),
                            stored_facts=(
                                UserDiscoveryStoredFact(
                                    fact_id="fact-basics",
                                    route_kind="user_profile",
                                    review_text="User prefers to be called Thom.",
                                    created_at=(now - timedelta(days=10)).isoformat(),
                                    updated_at=(now - timedelta(days=1)).isoformat(),
                                ),
                            ),
                        ),
                        UserDiscoveryTopicState(
                            topic_id="social",
                            last_answer_at=(now - timedelta(days=2)).isoformat(),
                            stored_facts=(
                                UserDiscoveryStoredFact(
                                    fact_id="fact-social",
                                    route_kind="contact",
                                    review_text="Anna is the user's daughter.",
                                    created_at=(now - timedelta(days=9)).isoformat(),
                                    updated_at=(now - timedelta(days=2)).isoformat(),
                                ),
                            ),
                        ),
                        UserDiscoveryTopicState(
                            topic_id="interests",
                            last_answer_at=(now - timedelta(days=4)).isoformat(),
                            stored_facts=(
                                UserDiscoveryStoredFact(
                                    fact_id="fact-interests",
                                    route_kind="preference",
                                    review_text="User prefers Melitta coffee.",
                                    created_at=(now - timedelta(days=8)).isoformat(),
                                    updated_at=(now - timedelta(days=4)).isoformat(),
                                ),
                            ),
                        ),
                    ),
                )
            )

            candidates = load_display_reserve_user_discovery_candidates(
                config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        context = candidates[0].generation_context or {}
        self.assertEqual(context.get("invite_kind"), "review_profile")
        self.assertEqual(context.get("topic_id"), "basics")

    def test_display_reserve_discovery_candidate_uses_human_prompt_anchor_not_raw_topic_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            now = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)

            candidates = load_display_reserve_user_discovery_candidates(
                config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        context = candidates[0].generation_context or {}
        self.assertEqual(context.get("topic_id"), "basics")
        self.assertEqual(context.get("display_anchor"), "dein Name")
        self.assertEqual(
            context.get("hook_hint"),
            "Ich moechte wissen, wie ich dich ansprechen soll.",
        )
        self.assertEqual(
            context.get("card_intent"),
            {
                "topic_semantics": "bevorzugte Anrede und Namensform",
                "statement_intent": "Twinr will wissen, wie es den Nutzer ansprechen soll.",
                "cta_intent": "Den Nutzer bitten, den passenden Namen oder die passende Anrede zu nennen.",
                "relationship_stance": "ruhiges Kennenlernen ohne Setup-Ton",
            },
        )
        self.assertNotIn("Basisinfos", str(context.get("hook_hint")))
        self.assertNotIn("Einrichtung", str(context.get("hook_hint")))

    def test_build_invitation_skips_basics_when_curated_user_profile_already_names_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("User: Thom.\nLives in Schwarzenbek.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            service = UserDiscoveryService.from_config(config)

            invite = service.build_invitation(now=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc))

        self.assertIsNotNone(invite)
        assert invite is not None
        self.assertEqual(invite.topic_id, "companion_style")
        self.assertEqual(invite.invite_kind, "start_setup")

    def test_display_reserve_candidates_skip_basics_when_curated_user_profile_already_names_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("User: Thom.\nLives in Schwarzenbek.\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            now = datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc)

            candidates = load_display_reserve_user_discovery_candidates(
                config,
                local_now=now,
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        context = candidates[0].generation_context or {}
        self.assertEqual(context.get("topic_id"), "companion_style")
        self.assertEqual(candidates[0].topic_key, "user_discovery:initial_setup:companion_style")

    def test_build_invitation_skips_basics_when_graph_preference_already_names_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            TwinrPersonalGraphStore.from_config(config).remember_preference(category="preferred_name", value="Thom")
            service = UserDiscoveryService.from_config(config)

            invite = service.build_invitation(now=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc))

        self.assertIsNotNone(invite)
        assert invite is not None
        self.assertEqual(invite.topic_id, "companion_style")

    def test_display_reserve_candidates_skip_basics_when_long_term_memory_already_names_user(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            LongTermStructuredStore.from_config(config).write_snapshot(
                objects=(
                    _active_preference_object(
                        memory_id="fact:user:main:prefers_name:thom",
                        summary="The user wants to be called Thom.",
                        attributes={
                            "subject_ref": "user:main",
                            "predicate": "prefers_name",
                            "preference_type": "name",
                            "preference_value": "Thom",
                            "support_count": 1,
                        },
                    ),
                ),
            )

            candidates = load_display_reserve_user_discovery_candidates(
                config,
                local_now=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc),
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        context = candidates[0].generation_context or {}
        self.assertEqual(context.get("topic_id"), "companion_style")
        self.assertEqual(candidates[0].topic_key, "user_discovery:initial_setup:companion_style")

    def test_build_invitation_skips_companion_style_when_authoritative_style_preference_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            TwinrPersonalGraphStore.from_config(config).remember_preference(category="preferred_name", value="Thom")
            LongTermStructuredStore.from_config(config).write_snapshot(
                objects=(
                    _active_preference_object(
                        memory_id="pref:initiative:gentle",
                        summary="The user likes gentle initiative in follow-ups.",
                        attributes={
                            "subject_ref": "user:main",
                            "memory_domain": "preference",
                            "predicate": "user_prefers_small_follow_up_when_helpful",
                            "preference_type": "initiative",
                            "preference_value": "gently_proactive",
                            "support_count": 1,
                        },
                    ),
                ),
            )
            service = UserDiscoveryService.from_config(config)

            invite = service.build_invitation(now=datetime(2026, 3, 26, 10, 0, tzinfo=timezone.utc))

        self.assertIsNotNone(invite)
        assert invite is not None
        self.assertEqual(invite.topic_id, "social")

    def test_runtime_answer_to_visible_discovery_card_retires_pending_reserve_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            personality_dir = Path(temp_dir) / "personality"
            personality_dir.mkdir(parents=True, exist_ok=True)
            (personality_dir / "USER.md").write_text("Base user profile\n", encoding="utf-8")
            (personality_dir / "PERSONALITY.md").write_text("Base personality\n", encoding="utf-8")
            config = TwinrConfig(
                project_root=temp_dir,
                personality_dir="personality",
                memory_markdown_path=str(Path(temp_dir) / "state" / "MEMORY.md"),
                runtime_state_path=str(Path(temp_dir) / "runtime-state.json"),
            )
            runtime = TwinrRuntime(config=config)
            history_store = DisplayAmbientImpulseHistoryStore.from_config(config)
            feedback_store = DisplayReserveBusFeedbackStore.from_config(config)
            shown_at = datetime(2026, 3, 26, 8, 59, tzinfo=timezone.utc)
            history_store.append_exposure(
                source="user_discovery",
                topic_key="user_discovery:initial_setup:basics",
                semantic_topic_key="user_discovery:initial_setup:basics",
                title="Basisinfos",
                headline="Ich moechte wissen, wie ich dich ansprechen soll.",
                body="Wenn du magst, kannst du es mir kurz sagen.",
                action="ask_one",
                attention_state="forming",
                shown_at=shown_at,
                expires_at=shown_at + timedelta(minutes=10),
                match_anchors=("Basisinfos", "dein Name"),
            )
            try:
                runtime.manage_user_discovery(
                    action="answer",
                    topic_id="basics",
                    learned_facts=(
                        UserDiscoveryFact(
                            storage="user_profile",
                            text="User prefers to be called Thom.",
                        ),
                    ),
                    topic_complete=True,
                    now=shown_at + timedelta(minutes=1),
                )
            finally:
                runtime.shutdown(timeout_s=0.1)

            history = history_store.load()
            reserve_feedback = feedback_store.load_active(now=shown_at + timedelta(minutes=1))

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].response_status, "engaged")
        self.assertEqual(history[0].response_mode, "voice_immediate_pickup")
        self.assertIsNotNone(reserve_feedback)
        assert reserve_feedback is not None
        self.assertEqual(reserve_feedback.topic_key, "user_discovery:initial_setup:basics")
        self.assertEqual(reserve_feedback.reaction, "immediate_engagement")

    def test_display_reserve_discovery_candidate_uses_follow_up_hook_after_partial_basics_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            now = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)
            service = UserDiscoveryService.from_config(config)
            callbacks = UserDiscoveryCommitCallbacks(
                update_user_profile=lambda category, instruction: None,
                delete_user_profile=lambda category: None,
                update_personality=lambda category, instruction: None,
                delete_personality=lambda category: None,
            )
            service.manage(action="start_or_resume", topic_id="basics", now=now)
            service.manage(
                action="answer",
                topic_id="basics",
                learned_facts=(
                    UserDiscoveryFact(
                        storage="user_profile",
                        text="User prefers to be called Thom.",
                    ),
                ),
                topic_complete=False,
                callbacks=callbacks,
                now=now + timedelta(minutes=1),
            )
            service.manage(
                action="pause_session",
                now=now + timedelta(minutes=2),
            )

            candidates = load_display_reserve_user_discovery_candidates(
                config,
                local_now=now + timedelta(hours=19),
                max_items=3,
            )

        self.assertEqual(len(candidates), 1)
        context = candidates[0].generation_context or {}
        self.assertEqual(context.get("topic_id"), "basics")
        self.assertEqual(context.get("display_prompt_stage"), "follow_up")
        self.assertNotEqual(
            context.get("hook_hint"),
            "Ich moechte wissen, wie ich dich ansprechen soll.",
        )

    def test_registry_exposes_manage_user_discovery_tool(self) -> None:
        self.assertIn("manage_user_discovery", realtime_tool_names())


if __name__ == "__main__":
    unittest.main()
