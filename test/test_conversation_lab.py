from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.agent.base_agent.contracts import ToolCallingTurnResponse
from twinr.agent.base_agent.state.snapshot import RuntimeSnapshotStore
from twinr.agent.base_agent.runtime.runtime import TwinrRuntime
from twinr.agent.tools.runtime.streaming_loop import ToolCallingStreamingLoop
from twinr.agent.tools.handlers.household_identity import handle_manage_household_identity
from twinr.agent.tools.handlers.output import handle_inspect_camera
from twinr.agent.tools.handlers.portrait_identity import handle_enroll_portrait_identity
from twinr.hardware.camera import CapturedPhoto
from twinr.memory.on_device import ConversationTurn
from twinr.ops.paths import resolve_ops_paths
from twinr.web.conversation_lab import (
    _CONVERSATION_LAB_TOOL_NAMES,
    _ConversationLabToolOwner,
    _build_tool_loop,
    _conversation_lab_runtime_config,
    _run_text_turn,
    _search_snapshot,
    run_conversation_lab_turn,
)


class _Collector:
    def __init__(self) -> None:
        self.emitted: list[str] = []
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self.usages: list[dict[str, object]] = []

    def emit(self, payload: str) -> None:
        self.emitted.append(payload)

    def record_event(self, event_name: str, message: str, **data: object) -> None:
        self.events.append((event_name, message, dict(data)))

    def record_usage(self, **data: object) -> None:
        self.usages.append(dict(data))


class _UsageStore:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def append(self, **data: object) -> None:
        self.rows.append(dict(data))


class _RuntimeStub:
    def __init__(self) -> None:
        self.user_voice_status = "portal_operator_authenticated"
        self.memory = SimpleNamespace(remember=lambda *_args, **_kwargs: None)
        self.long_term_memory = SimpleNamespace(writer=None, multimodal_writer=None)
        self.finalized_answers: list[str] = []
        self.flush_calls: list[float] = []
        self.recorded_tool_history: list[tuple[tuple[object, ...], tuple[object, ...]]] = []
        self.shutdown_calls: list[float] = []
        self.tool_context: tuple[tuple[str, str], ...] = ()

    def apply_live_config(self, updated_config: TwinrConfig) -> None:
        self.config = updated_config

    def provider_conversation_context(self):
        return ()

    def tool_provider_conversation_context(self):
        return self.tool_context

    def supervisor_provider_conversation_context(self):
        return self.tool_context

    def tool_provider_text_surface_conversation_context(self):
        return self.tool_context

    def supervisor_provider_text_surface_conversation_context(self):
        return self.tool_context

    def finalize_agent_turn(self, answer: str) -> str:
        self.finalized_answers.append(answer)
        return answer

    def flush_long_term_memory(self, *, timeout_s: float = 2.0) -> bool:
        self.flush_calls.append(timeout_s)
        return True

    def record_personality_tool_history(self, *, tool_calls, tool_results) -> None:
        self.recorded_tool_history.append((tuple(tool_calls), tuple(tool_results)))

    def shutdown(self, timeout_s: float = 2.0) -> None:
        self.shutdown_calls.append(timeout_s)


class _VisionPrintBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def respond_to_images_with_metadata(
        self,
        prompt: str,
        *,
        images,
        conversation,
        allow_web_search: bool,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "images": list(images),
                "conversation": conversation,
                "allow_web_search": allow_web_search,
            }
        )
        return SimpleNamespace(
            text="Ich sehe ein einzelnes Gesicht.",
            response_id="resp_vision_1",
            request_id="req_vision_1",
            model="gpt-test",
            token_usage=None,
        )


class _LoopProvider:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []

    def start_turn_streaming(
        self,
        prompt: str,
        *,
        conversation=None,
        instructions=None,
        tool_schemas=(),
        allow_web_search=None,
        on_text_delta=None,
    ) -> ToolCallingTurnResponse:
        del instructions, tool_schemas, allow_web_search, on_text_delta
        self.start_calls.append({"prompt": prompt, "conversation": tuple(conversation or ())})
        return ToolCallingTurnResponse(
            text="In New York ist es gerade 10:54 Uhr.",
            tool_calls=(),
            continuation_token=None,
            response_id="resp_text_turn",
            request_id="req_text_turn",
            model="gpt-test",
            used_web_search=False,
        )


class ConversationLabToolOwnerTests(unittest.TestCase):
    def _make_owner(self, *, project_root: Path, print_backend: object | None = None) -> _ConversationLabToolOwner:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=project_root,
            personality_dir="personality",
            camera_device="/dev/video0",
            vision_reference_image_path=None,
        )
        env_path = project_root / ".env"
        env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
        return _ConversationLabToolOwner(
            config=config,
            env_path=env_path,
            runtime=_RuntimeStub(),
            print_backend=print_backend,
            usage_store=_UsageStore(),
            collector=_Collector(),
            configurable_providers=(),
        )

    def test_conversation_lab_exposes_portrait_and_camera_tools(self) -> None:
        self.assertIn("browser_automation", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("list_smart_home_entities", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("read_smart_home_state", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("control_smart_home_entities", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("read_smart_home_sensor_stream", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("enroll_portrait_identity", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("get_portrait_identity_status", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("reset_portrait_identity", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("manage_household_identity", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("inspect_camera", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("manage_voice_quiet_mode", _CONVERSATION_LAB_TOOL_NAMES)

    def test_owner_supports_portrait_identity_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = self._make_owner(project_root=Path(temp_dir))
            fake_camera = object()
            fake_provider = SimpleNamespace(
                capture_and_enroll_reference=lambda **_kwargs: SimpleNamespace(
                    status="enrolled",
                    user_id="main_user",
                    display_name="Theo",
                    reference_id="ref_local_1",
                    reference_image_count=1,
                )
            )
            with (
                patch("twinr.web.conversation_lab_vision.V4L2StillCamera.from_config", return_value=fake_camera),
                patch("twinr.agent.tools.handlers.portrait_identity.PortraitMatchProvider.from_config", return_value=fake_provider),
            ):
                result = handle_enroll_portrait_identity(owner, {"display_name": "Theo"})

            self.assertEqual(result["status"], "enrolled")
            self.assertEqual(result["reference_image_count"], 1)
            self.assertIs(owner.camera, fake_camera)

    def test_owner_supports_household_identity_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            owner = self._make_owner(project_root=Path(temp_dir))
            fake_camera = object()
            fake_manager = SimpleNamespace(
                primary_user_id="main_user",
                status=lambda **_kwargs: SimpleNamespace(
                    primary_user_id="main_user",
                    members=(),
                    current_observation=None,
                ),
            )
            with (
                patch("twinr.web.conversation_lab_vision.V4L2StillCamera.from_config", return_value=fake_camera),
                patch("twinr.agent.tools.handlers.household_identity.HouseholdIdentityManager.from_config", return_value=fake_manager),
            ):
                result = handle_manage_household_identity(owner, {"action": "status"})

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["member_count"], 0)
            self.assertIs(owner.camera, fake_camera)

    def test_owner_supports_inspect_camera_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            print_backend = _VisionPrintBackend()
            owner = self._make_owner(project_root=Path(temp_dir), print_backend=print_backend)
            fake_camera = SimpleNamespace(
                capture_photo=lambda *, filename: CapturedPhoto(
                    data=b"\x89PNG\r\n\x1a\nfake-png-bytes",
                    content_type="image/png",
                    filename=filename,
                    source_device="/dev/video0",
                    input_format="yuyv422",
                )
            )
            with patch("twinr.web.conversation_lab_vision.V4L2StillCamera.from_config", return_value=fake_camera):
                result = handle_inspect_camera(owner, {"question": "Was siehst du?"})

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["answer"], "Ich sehe ein einzelnes Gesicht.")
            self.assertEqual(len(print_backend.calls), 1)
            self.assertEqual(len(print_backend.calls[0]["images"]), 1)
            self.assertIn("This request includes camera input.", print_backend.calls[0]["prompt"])

    def test_search_snapshot_returns_timeout_panel_when_operator_search_exceeds_budget(self) -> None:
        config = TwinrConfig(project_root=".")

        def _slow_search(*_args, **_kwargs):
            time.sleep(0.05)
            return object()

        with (
            patch("twinr.web.conversation_lab._CONVERSATION_LAB_SEARCH_TIMEOUT_S", 0.01),
            patch("twinr.web.conversation_lab.run_long_term_operator_search", side_effect=_slow_search),
        ):
            panel = _search_snapshot(config, "Wie ist der Status im Haus?")

        self.assertIn("TimeoutError", str(panel.get("status", {}).get("detail") or ""))

    def test_conversation_lab_runtime_config_disables_background_turn_writers(self) -> None:
        config = TwinrConfig(
            project_root=".",
            adaptive_timing_enabled=True,
            long_term_memory_background_store_turns=True,
            restore_runtime_state_on_startup=True,
            runtime_state_path="state/runtime-state.json",
        )

        runtime_config = _conversation_lab_runtime_config(
            config,
            session_id="session_20260329T201407Z_d4f218cc",
        )

        self.assertFalse(runtime_config.adaptive_timing_enabled)
        self.assertFalse(runtime_config.long_term_memory_background_store_turns)
        self.assertFalse(runtime_config.restore_runtime_state_on_startup)
        self.assertIn(
            "runtime-scopes/conversation-lab-session-20260329t201407z-d4f218cc/runtime-state.json",
            runtime_config.runtime_state_path,
        )

    def test_conversation_lab_runtime_config_ignores_shared_runtime_snapshot_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shared_runtime_state = root / "state" / "runtime-state.json"
            shared_runtime_state.parent.mkdir(parents=True, exist_ok=True)
            RuntimeSnapshotStore(shared_runtime_state).save(
                status="waiting",
                memory_turns=(
                    ConversationTurn(
                        "user",
                        "Wie spaet ist es in New York?",
                        datetime(2026, 3, 29, 20, 14, tzinfo=timezone.utc),
                    ),
                    ConversationTurn(
                        "assistant",
                        "In New York ist es gerade 16:14 Uhr.",
                        datetime(2026, 3, 29, 20, 14, 5, tzinfo=timezone.utc),
                    ),
                ),
                last_transcript="Wie spaet ist es in New York?",
                last_response="In New York ist es gerade 16:14 Uhr.",
            )
            config = TwinrConfig(
                project_root=str(root),
                runtime_state_path="state/runtime-state.json",
                restore_runtime_state_on_startup=True,
                long_term_memory_enabled=False,
            )

            runtime = TwinrRuntime(
                _conversation_lab_runtime_config(
                    config,
                    session_id="session_20260329T201602Z_2ff04632",
                )
            )
            try:
                self.assertEqual(tuple(runtime.memory.turns), ())
            finally:
                runtime.shutdown(timeout_s=0.1)

    def test_build_tool_loop_filters_conversation_lab_tools_by_runtime_availability(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
            llm_provider="anthropic",
        )
        owner = SimpleNamespace(
            print_backend=None,
            _configurable_providers=(),
            collector=SimpleNamespace(trace_event=lambda *args, **kwargs: None, trace_decision=lambda *args, **kwargs: None),
        )
        provider_bundle = SimpleNamespace(
            print_backend=None,
            tool_agent=object(),
            support_backend=object(),
        )
        captured: dict[str, object] = {}

        def _fake_loop(*, provider, tool_handlers, tool_schemas, stream_final_only):
            captured["provider"] = provider
            captured["tool_handlers"] = dict(tool_handlers)
            captured["tool_schemas"] = list(tool_schemas)
            captured["stream_final_only"] = stream_final_only
            return "loop-sentinel"

        with (
            patch("twinr.web.conversation_lab.build_streaming_provider_bundle", return_value=provider_bundle),
            patch("twinr.web.conversation_lab.RealtimeToolExecutor", return_value=object()),
            patch(
                "twinr.web.conversation_lab.available_realtime_tool_names",
                return_value=("search_live_info", "browser_automation"),
            ),
            patch(
                "twinr.web.conversation_lab.bind_realtime_tool_handlers",
                return_value={
                    "search_live_info": object(),
                    "browser_automation": object(),
                    "schedule_reminder": object(),
                },
            ),
            patch(
                "twinr.web.conversation_lab.build_agent_tool_schemas",
                side_effect=lambda tool_names: [{"type": "function", "name": name} for name in tool_names],
            ),
            patch("twinr.web.conversation_lab.ToolCallingStreamingLoop", side_effect=_fake_loop),
        ):
            loop, _resources = _build_tool_loop(config=config, owner=owner)

        self.assertEqual(loop, "loop-sentinel")
        self.assertEqual(tuple(captured["tool_handlers"].keys()), ("search_live_info", "browser_automation"))
        self.assertEqual(
            [schema["name"] for schema in captured["tool_schemas"]],
            ["search_live_info", "browser_automation"],
        )

    def test_run_conversation_lab_turn_skips_flush_and_personality_history_without_background_writers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            env_path = project_root / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            config = TwinrConfig(
                openai_api_key="test-key",
                project_root=project_root,
                personality_dir="personality",
            )
            ops_paths = resolve_ops_paths(project_root)
            runtime = _RuntimeStub()
            snapshot = SimpleNamespace(
                profile="tool",
                source="built_sync",
                query_profile=SimpleNamespace(
                    original_text="Wie ist der Status im Haus?",
                    retrieval_text="Wie ist der Status im Haus?",
                    canonical_english_text="How is the status in the house?",
                ),
                context=SimpleNamespace(
                    subtext_context=None,
                    topic_context=None,
                    midterm_context=None,
                    durable_context="durable:{status: ruhig}",
                    episodic_context="episodic:{user asked for house status}",
                    graph_context=None,
                    conflict_context=None,
                ),
            )
            runtime.long_term_memory = SimpleNamespace(
                writer=None,
                multimodal_writer=None,
                latest_context_snapshot=lambda *, profile: snapshot if profile == "tool" else None,
            )
            result = SimpleNamespace(
                text="Im Haus ist aktuell ruhig.",
                tool_calls=(),
                tool_results=(),
                model="gpt-test",
                response_id="resp_1",
                request_id="req_1",
                used_web_search=False,
                token_usage=None,
            )

            with (
                patch("twinr.web.conversation_lab.TwinrRuntime", return_value=runtime),
                patch("twinr.web.conversation_lab.TwinrUsageStore.from_config", return_value=_UsageStore()),
                patch("twinr.web.conversation_lab._build_tool_loop", return_value=(object(), ())),
                patch("twinr.web.conversation_lab._run_text_turn", return_value=result),
                patch("twinr.web.conversation_lab._search_snapshot", side_effect=AssertionError("_search_snapshot should not run")),
            ):
                session_id = run_conversation_lab_turn(
                    config,
                    env_path,
                    ops_paths,
                    session_id=None,
                    prompt="Wie ist der Status im Haus?",
                )

            session_path = ops_paths.ops_store_root / "conversation_lab" / f"{session_id}.json"
            payload = session_path.read_text(encoding="utf-8")
            parsed = json.loads(payload)
            turn = parsed["turns"][-1]

            self.assertIn('"status": "ok"', payload)
            self.assertIn('"Flush result"', payload)
            self.assertIn("Captured Context Before Answer", payload)
            self.assertIn("Post-Turn Context", payload)
            self.assertIn("durable:{status: ruhig}", payload)
            self.assertEqual(runtime.flush_calls, [])
            self.assertEqual(runtime.recorded_tool_history, [])
            self.assertEqual(runtime.shutdown_calls, [2.0])
            self.assertTrue(
                any(
                    item.get("title") == "Session Runtime Scope"
                    and "runtime_snapshot_scope: session_scoped" in tuple(item.get("meta_lines", ()))
                    and "restore_runtime_state_on_startup: false" in tuple(item.get("meta_lines", ()))
                    for item in turn.get("telemetry_items", [])
                )
            )

    def test_run_text_turn_injects_recent_thread_carryover_guidance(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        runtime = _RuntimeStub()
        runtime.tool_context = (
            ("system", "memory summary"),
            ("user", "Wie spaet ist es in New York?"),
            ("assistant", "In New York ist es gerade 10:53 Uhr."),
        )
        provider = _LoopProvider()
        loop = ToolCallingStreamingLoop(
            provider=provider,
            tool_handlers={},
            tool_schemas=(),
        )

        _run_text_turn(
            config=config,
            runtime=runtime,
            loop=loop,
            prompt="Ich meinte, wie spaet es ist.",
        )

        conversation = provider.start_calls[0]["conversation"]
        assert isinstance(conversation, tuple)
        self.assertTrue(
            any(
                role == "system"
                and "Recent thread carryover for this turn." in content
                and "New York" in content
                for role, content in conversation
            )
        )

    def test_run_text_turn_uses_recent_thread_rewritten_prompt(self) -> None:
        config = TwinrConfig(
            openai_api_key="test-key",
            project_root=".",
            personality_dir="personality",
        )
        runtime = _RuntimeStub()
        runtime.tool_context = (
            ("user", "Wie spaet ist es in New York?"),
            ("assistant", "In New York ist es gerade 10:53 Uhr."),
        )
        provider = _LoopProvider()
        loop = ToolCallingStreamingLoop(
            provider=provider,
            tool_handlers={},
            tool_schemas=(),
        )

        with patch(
            "twinr.web.conversation_lab.maybe_rewrite_prompt_against_recent_thread",
            return_value=SimpleNamespace(
                original_prompt="Ich meinte, wie spaet es ist.",
                effective_prompt="Wie spaet ist es in New York?",
                resolution="rewrite",
            ),
        ):
            _run_text_turn(
                config=config,
                runtime=runtime,
                loop=loop,
                prompt="Ich meinte, wie spaet es ist.",
            )

        self.assertEqual(provider.start_calls[0]["prompt"], "Wie spaet ist es in New York?")


if __name__ == "__main__":
    unittest.main()
