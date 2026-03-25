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
from twinr.agent.base_agent import TwinrConfig
from twinr.display.ambient_impulse_history import DisplayAmbientImpulseHistoryStore
from twinr.memory.user_discovery import (
    UserDiscoveryFact,
    UserDiscoveryMemoryRoute,
    UserDiscoveryService,
    UserDiscoveryState,
    UserDiscoveryStoredFact,
    UserDiscoveryTopicState,
)
from twinr.proactive.runtime.display_reserve_user_discovery import load_display_reserve_user_discovery_candidates
from twinr.agent.base_agent import TwinrRuntime


class UserDiscoveryTests(unittest.TestCase):
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
        self.assertEqual(candidates[0].generation_context.get("invite_kind"), "review_profile")
        self.assertEqual(candidates[0].generation_context.get("topic_id"), "basics")

    def test_registry_exposes_manage_user_discovery_tool(self) -> None:
        self.assertIn("manage_user_discovery", realtime_tool_names())


if __name__ == "__main__":
    unittest.main()
