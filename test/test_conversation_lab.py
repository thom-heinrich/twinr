from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.agent.tools.handlers.household_identity import handle_manage_household_identity
from twinr.agent.tools.handlers.output import handle_inspect_camera
from twinr.agent.tools.handlers.portrait_identity import handle_enroll_portrait_identity
from twinr.hardware.camera import CapturedPhoto
from twinr.web.conversation_lab import _CONVERSATION_LAB_TOOL_NAMES, _ConversationLabToolOwner


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

    def apply_live_config(self, updated_config: TwinrConfig) -> None:
        self.config = updated_config

    def provider_conversation_context(self):
        return ()


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


if __name__ == "__main__":
    unittest.main()
