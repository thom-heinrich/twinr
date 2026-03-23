from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.tools.prompting.instructions import (
    build_compact_tool_agent_instructions,
    build_first_word_instructions,
    build_local_route_first_word_instructions,
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
        self.assertIn("manage_household_identity", instructions)
        self.assertIn("enroll_portrait_identity tool", instructions)
        self.assertIn("guidance_hints", instructions)
        self.assertIn("Repeated enroll_portrait_identity calls", instructions)
        self.assertIn("configure_world_intelligence tool", instructions)
        self.assertIn("occasional recalibration", instructions)
        self.assertIn("manage_user_discovery", instructions)
        self.assertIn("guided get-to-know-you flow", instructions)
        self.assertIn("list_smart_home_entities", instructions)
        self.assertIn("control_smart_home_entities", instructions)
        self.assertIn("broad smart-home or house-status question", instructions)
        self.assertIn("state_filters", instructions)
        self.assertIn("aggregate_by", instructions)
        self.assertIn("build a small live situation picture", instructions)
        self.assertIn("two to four targeted smart-home queries", instructions)
        self.assertIn("copy the exact routed entity_id values verbatim", instructions)
        self.assertIn("catch-all entity dump", instructions)
        self.assertIn("list is truncated", instructions)
        self.assertIn("do not treat that alone as the whole house status", instructions)

    def test_compact_tool_instructions_cover_portrait_identity_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_compact_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("Use enroll_portrait_identity", instructions)
        self.assertIn("Prefer manage_household_identity", instructions)
        self.assertIn("guidance_hints and recommended_next_step", instructions)
        self.assertIn("read_smart_home_sensor_stream", instructions)
        self.assertIn("grouped smart-home counts", instructions)
        self.assertIn("multiple targeted smart-home queries", instructions)
        self.assertIn("aggregate_by first", instructions)
        self.assertIn("copied verbatim from prior smart-home tool results", instructions)
        self.assertIn("catch-all entity list", instructions)
        self.assertIn("manage_user_discovery", instructions)
        self.assertIn("sensitive permission gate", instructions)

    def test_discovery_instructions_cover_semantic_intents_and_self_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("Treat discovery intents semantically", instructions)
        self.assertIn("they want Twinr to know them better", instructions)
        self.assertIn("already authorizes start_or_resume", instructions)
        self.assertIn("The start request itself is the confirmation", instructions)
        self.assertIn("freely volunteers a stable self-detail", instructions)
        self.assertIn("action answer directly", instructions)
        self.assertIn("Prefer memory_routes over learned_facts", instructions)
        self.assertIn("daughter, son, friend, or caregiver details", instructions)
        self.assertIn("Each distinct learned detail should become its own learned_fact or memory_route", instructions)
        self.assertIn("Do not merge a contact, a preference, and a future plan into one combined route", instructions)
        self.assertIn("Do not invent placeholder fact_id values", instructions)
        self.assertIn("only use a fact_id that actually came back from review_profile", instructions)
        self.assertIn("direct first-person profile statement", instructions)
        self.assertIn("Imperative wording about how Twinr should address the user", instructions)
        self.assertIn("Names, preferred forms of address, family relations, and favorite brands", instructions)
        self.assertIn("not a request for permission to store it", instructions)
        self.assertIn("rename instruction itself as the confirmation", instructions)
        self.assertIn("Map semantic start wording to start_or_resume", instructions)
        self.assertIn("do not ask a second permission question", instructions)
        self.assertIn("what have you learned about me", instructions)
        self.assertIn("replace_fact or delete_fact in the same turn", instructions)
        self.assertIn("special setup phrase", instructions)

    def test_compact_discovery_instructions_cover_semantic_intents_and_self_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_compact_tool_agent_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("Treat discovery semantically", instructions)
        self.assertIn("freely volunteers stable profile details", instructions)
        self.assertIn("already authorizes start discovery", instructions)
        self.assertIn("The start request itself is the confirmation", instructions)
        self.assertIn("action answer directly", instructions)
        self.assertIn("Prefer memory_routes when the user names a person and relationship", instructions)
        self.assertIn("Use one learned_fact or memory_route per distinct learned detail", instructions)
        self.assertIn("Do not invent placeholder fact_id values", instructions)
        self.assertIn("Direct first-person profile or preference statements", instructions)
        self.assertIn("Imperative wording about how Twinr should address the user", instructions)
        self.assertIn("Names, preferred forms of address, family relations, and favorite brands", instructions)
        self.assertIn("not a request for permission to store it", instructions)
        self.assertIn("rename instruction itself as the confirmation", instructions)
        self.assertIn("Map semantic start wording to start discovery", instructions)
        self.assertIn("do not ask a second permission question", instructions)
        self.assertIn("stored profile details", instructions)
        self.assertIn("special setup phrase", instructions)

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
        self.assertIn("leave spoken_ack empty", instructions)
        self.assertIn("Do not wait for the specialist result", instructions)
        self.assertIn("Open smart-home or house-status questions usually need handoff_specialist_worker", instructions)

    def test_supervisor_decision_instructions_use_structured_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            instructions = build_supervisor_decision_instructions(
                TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            )

        self.assertIn("three structured actions", instructions)
        self.assertIn("Choose handoff", instructions)
        self.assertIn("Choose handoff for open smart-home or house-status questions", instructions)
        self.assertIn("spoken_ack is optional", instructions)
        self.assertIn("leave spoken_ack null", instructions)
        self.assertIn("generic stock phrase", instructions)
        self.assertIn("must not imply the task is already finished", instructions)
        self.assertIn("short conversational follow-up", instructions)
        self.assertIn("calm companion voice", instructions)
        self.assertIn("context_scope", instructions)
        self.assertIn("location_hint", instructions)
        self.assertIn("date_context", instructions)
        self.assertIn("plain spoken language only", instructions)
        self.assertIn("no markdown", instructions)
        self.assertIn("broader memory", instructions)
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

    def test_local_route_first_word_overlay_forces_provisional_route_specific_filler(self) -> None:
        instructions = build_local_route_first_word_instructions(
            "memory",
            handoff_goal="Answer using the user's persisted or recent Twinr memory.",
            language_hint="de",
        )

        self.assertIn("slower specialist lane", instructions)
        self.assertIn("Do not answer the user's question directly", instructions)
        self.assertIn("Return mode filler only", instructions)
        self.assertIn("recalling or checking remembered details", instructions)
        self.assertIn("natural German", instructions)
        self.assertIn("persisted or recent Twinr memory", instructions)

    def test_fast_lane_instructions_allow_stable_general_knowledge_direct_replies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(openai_api_key="test-key", project_root=temp_dir, personality_dir="personality")
            first_word_instructions = build_first_word_instructions(config)
            supervisor_tool_instructions = build_supervisor_tool_agent_instructions(config)
            supervisor_decision_instructions = build_supervisor_decision_instructions(config)

        self.assertIn("built-in model knowledge", first_word_instructions)
        self.assertIn("stable non-fresh explainers", first_word_instructions)
        self.assertIn("built-in model knowledge", supervisor_tool_instructions)
        self.assertIn("everyday how or why questions", supervisor_tool_instructions)
        self.assertIn("built-in model knowledge", supervisor_decision_instructions)
        self.assertIn("everyday how or why questions", supervisor_decision_instructions)


if __name__ == "__main__":
    unittest.main()
