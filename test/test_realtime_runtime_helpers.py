"""Targeted regression tests for extracted realtime runtime helper modules."""

import importlib
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import tempfile
import unittest
from zoneinfo import ZoneInfo

from twinr.agent.workflows.realtime_runtime.background_delivery import (
    BackgroundDeliveryBlocked,
    begin_background_delivery,
)
from twinr.agent.workflows.realtime_runtime.reminder_delivery import (
    LocalMetadataResponse,
    phrase_due_reminder_with_fallback,
)
from twinr.agent.workflows.realtime_runtime.vision_support import (
    build_vision_prompt,
    load_reference_image,
)


class _BackgroundLoopHarness:
    def __init__(self, *, status: str = "waiting", conversation_active: bool = False) -> None:
        self.runtime = SimpleNamespace(status=SimpleNamespace(value=status))
        self._conversation_session_active = conversation_active
        self._lock = RLock()

    def _get_lock(self, _name: str) -> RLock:
        return self._lock


class _ReminderLoopHarness:
    def __init__(self) -> None:
        self.emitted: list[str] = []
        self.events: list[tuple[str, str, dict[str, object]]] = []
        self.agent_provider = SimpleNamespace(
            phrase_due_reminder_with_metadata=self._primary_backend,
            respond_with_metadata=self._generic_backend,
        )

    def _primary_backend(self, _reminder):
        raise RuntimeError("primary down")

    def _generic_backend(self, *_args, **_kwargs):
        return SimpleNamespace(text="")

    def _coerce_text(self, value) -> str:
        return str(value or "").strip()

    def _safe_emit(self, message: str) -> None:
        self.emitted.append(message)

    def _safe_record_event(self, event: str, message: str, **kwargs: object) -> None:
        self.events.append((event, message, kwargs))

    def _local_timezone(self) -> ZoneInfo:
        return ZoneInfo("UTC")

    def _local_timezone_name(self) -> str:
        return "UTC"

    def _remember_background_fault(self, *_args, **_kwargs) -> None:
        return None


class _VisionLoopHarness:
    def __init__(self, *, reference_path: str, base_dir: str) -> None:
        self.config = SimpleNamespace(
            vision_reference_image_path=reference_path,
            vision_reference_image_base_dir=base_dir,
            vision_reference_image_max_bytes=1024,
        )
        self.emitted: list[str] = []

    def _try_emit(self, message: str) -> None:
        self.emitted.append(message)

    def _safe_error_text(self, exc: BaseException) -> str:
        return str(exc)


class BackgroundDeliveryHelpersTests(unittest.TestCase):
    def test_begin_background_delivery_runs_action_when_idle(self) -> None:
        loop = _BackgroundLoopHarness()

        result = begin_background_delivery(loop, lambda: "ok")

        self.assertEqual(result, "ok")

    def test_begin_background_delivery_blocks_when_runtime_is_busy(self) -> None:
        loop = _BackgroundLoopHarness(status="answering")

        with self.assertRaises(BackgroundDeliveryBlocked) as ctx:
            begin_background_delivery(loop, lambda: "nope")

        self.assertEqual(ctx.exception.reason, "busy")


class ReminderDeliveryHelpersTests(unittest.TestCase):
    def test_phrase_due_reminder_with_fallback_returns_local_metadata_response(self) -> None:
        loop = _ReminderLoopHarness()
        reminder = SimpleNamespace(
            reminder_id="rem-1",
            summary="Take your medicine",
            details=None,
            original_request=None,
            due_at=None,
            kind="medication",
        )

        response = phrase_due_reminder_with_fallback(
            loop,
            reminder,
            instructions="Speak the reminder now.",
        )

        self.assertIsInstance(response, LocalMetadataResponse)
        self.assertEqual(response.text, "Reminder. Take your medicine")
        self.assertIn("reminder_backend_primary_error=primary down", loop.emitted)
        self.assertIn("reminder_backend_fallback=generic", loop.emitted)
        self.assertIn("reminder_backend_fallback=local", loop.emitted)


class VisionSupportHelpersTests(unittest.TestCase):
    def test_build_vision_prompt_mentions_reference_image_when_requested(self) -> None:
        prompt = build_vision_prompt("Bin ich das?", include_reference=True)

        self.assertIn("Image 2 is a stored reference image of the main user.", prompt)
        self.assertIn("User request: Bin ich das?", prompt)

    def test_load_reference_image_rejects_paths_outside_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_dir = temp_path / "safe"
            base_dir.mkdir()
            reference_path = temp_path / "outside.jpg"
            reference_path.write_bytes(b"\xff\xd8\xfftest")
            loop = _VisionLoopHarness(
                reference_path=str(reference_path),
                base_dir=str(base_dir),
            )

            result = load_reference_image(
                loop,
                allowed_suffixes=frozenset({".jpg"}),
                default_max_bytes=1024,
            )

        self.assertIsNone(result)
        self.assertIn(
            "vision_reference_rejected=outside_base_dir:outside.jpg",
            loop.emitted,
        )


class WorkflowPackageExportsTests(unittest.TestCase):
    def test_workflow_package_dir_lists_only_active_loop_exports(self) -> None:
        module = importlib.import_module("twinr.agent.workflows")

        self.assertIn("TwinrRealtimeHardwareLoop", module.__dir__())
        self.assertIn("TwinrStreamingHardwareLoop", module.__dir__())
        self.assertNotIn("TwinrHardwareLoop", module.__dir__())


if __name__ == "__main__":
    unittest.main()
