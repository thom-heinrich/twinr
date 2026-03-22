from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.tools.handlers.household_identity import handle_manage_household_identity
from twinr.agent.tools.handlers.output import handle_inspect_camera
from twinr.agent.tools.handlers.portrait_identity import handle_enroll_portrait_identity
from twinr.hardware.camera import CapturedPhoto
from twinr.ops.paths import resolve_ops_paths
from twinr.web.conversation_lab import (
    _CONVERSATION_LAB_TOOL_NAMES,
    _ConversationLabToolOwner,
    _conversation_lab_runtime_config,
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

    def apply_live_config(self, updated_config: TwinrConfig) -> None:
        self.config = updated_config

    def provider_conversation_context(self):
        return ()

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
        self.assertIn("list_smart_home_entities", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("read_smart_home_state", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("control_smart_home_entities", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("read_smart_home_sensor_stream", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("enroll_portrait_identity", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("get_portrait_identity_status", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("reset_portrait_identity", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("manage_household_identity", _CONVERSATION_LAB_TOOL_NAMES)
        self.assertIn("inspect_camera", _CONVERSATION_LAB_TOOL_NAMES)

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
                    data=b"not-really-a-png",
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
        )

        runtime_config = _conversation_lab_runtime_config(config)

        self.assertFalse(runtime_config.adaptive_timing_enabled)
        self.assertFalse(runtime_config.long_term_memory_background_store_turns)

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
                patch("twinr.web.conversation_lab._search_snapshot", return_value={"status": {"detail": "ok"}}),
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

            self.assertIn('"status": "ok"', payload)
            self.assertIn('"Flush result"', payload)
            self.assertEqual(runtime.flush_calls, [])
            self.assertEqual(runtime.recorded_tool_history, [])
            self.assertEqual(runtime.shutdown_calls, [2.0])


if __name__ == "__main__":
    unittest.main()
