from __future__ import annotations

from pathlib import Path
from queue import Queue
from threading import Lock
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.realtime_runtime.background import TwinrRealtimeBackgroundMixin
from twinr.agent.workflows.realtime_runtime.support import TwinrRealtimeSupportMixin
from twinr.config import TwinrConfig
from twinr.runtime import TwinrRuntime
from twinr.memory.longterm.runtime.service import LongTermMemoryService


class _FakeCameraCapture:
    def __init__(self) -> None:
        self.source_device = "/dev/video0"
        self.input_format = "mjpeg"
        self.data = b"\x89PNG\r\n\x1a\ncamera"
        self.content_type = "image/png"
        self.filename = "camera-capture.png"


class _FakeCamera:
    def __init__(self) -> None:
        self.capture_calls = 0

    def capture_photo(self, *, filename: str = "camera-capture.png", output_path=None):
        self.capture_calls += 1
        return _FakeCameraCapture()


class _BackgroundHarness(TwinrRealtimeBackgroundMixin):
    def __init__(self, runtime: TwinrRuntime) -> None:
        self.runtime = runtime
        self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue()


class _SupportHarness(TwinrRealtimeSupportMixin):
    def __init__(self, runtime: TwinrRuntime, camera: _FakeCamera, config: TwinrConfig) -> None:
        self.runtime = runtime
        self.camera = camera
        self.config = config
        self._camera_lock = Lock()
        self.emit = lambda _line: None


def _config(root: str) -> TwinrConfig:
    return TwinrConfig(
        project_root=root,
        personality_dir="personality",
        memory_markdown_path=str(Path(root) / "state" / "MEMORY.md"),
        long_term_memory_enabled=True,
        long_term_memory_write_queue_size=8,
        long_term_memory_path=str(Path(root) / "state" / "chonkydb"),
    )


def _sensor_facts() -> dict[str, object]:
    return {
        "pir": {
            "motion_detected": True,
            "low_motion": False,
            "no_motion_for_s": 0.0,
        },
        "camera": {
            "person_visible": True,
            "person_visible_for_s": 12.0,
            "looking_toward_device": True,
            "body_pose": "standing",
            "smiling": False,
            "hand_or_object_near_camera": True,
            "hand_or_object_near_camera_for_s": 4.0,
        },
        "vad": {
            "speech_detected": False,
            "speech_detected_for_s": 0.0,
            "quiet": True,
            "quiet_for_s": 9.0,
            "distress_detected": False,
        },
    }


class LongTermMultimodalTests(unittest.TestCase):
    def test_service_can_promote_repeated_sensor_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))

            for _ in range(2):
                service.enqueue_multimodal_evidence(
                    event_name="sensor_observation",
                    modality="sensor",
                    source="proactive_monitor",
                    message="Changed multimodal sensor observation recorded.",
                    data={
                        "facts": _sensor_facts(),
                        "event_names": ["pir.motion_detected", "camera.person_visible"],
                    },
                )
            service.flush(timeout_s=2.0)
            objects = {item.memory_id: item for item in service.object_store.load_objects()}
            service.shutdown()

        presence = next(
            item
            for item in objects.values()
            if item.kind == "pattern" and (item.attributes or {}).get("pattern_type") == "presence"
        )
        interaction = next(
            item for item in objects.values() if item.memory_id.startswith("pattern:camera_interaction:")
        )
        self.assertEqual(presence.status, "active")
        self.assertEqual(interaction.status, "active")
        self.assertGreaterEqual((presence.attributes or {}).get("support_count", 0), 2)
        self.assertTrue(any(item.kind == "observation" for item in objects.values()))

    def test_runtime_buttons_enqueue_multimodal_button_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=_config(temp_dir))

            runtime.press_green_button()
            runtime.cancel_listening()
            runtime.last_response = "Printed answer."
            runtime.press_yellow_button()
            runtime.flush_long_term_memory(timeout_s=2.0)
            objects = tuple(runtime.long_term_memory.object_store.load_objects())
            runtime.shutdown(timeout_s=2.0)

        summaries = [
            item.summary
            for item in objects
            if item.kind == "pattern" and (item.attributes or {}).get("pattern_type") == "interaction"
        ]
        self.assertTrue(any("green button" in summary for summary in summaries))
        self.assertTrue(any("yellow button" in summary for summary in summaries))

    def test_background_sensor_handler_enqueues_multimodal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=_config(temp_dir))
            harness = _BackgroundHarness(runtime)

            harness.handle_sensor_observation(_sensor_facts(), ("pir.motion_detected", "camera.person_visible"))
            queued_facts, queued_events = harness._sensor_observation_queue.get_nowait()
            runtime.flush_long_term_memory(timeout_s=2.0)
            objects = tuple(runtime.long_term_memory.object_store.load_objects())
            runtime.shutdown(timeout_s=2.0)

        self.assertIn("pir", queued_facts)
        self.assertEqual(queued_events, ("pir.motion_detected", "camera.person_visible"))
        self.assertTrue(
            any(
                item.kind == "pattern" and (item.attributes or {}).get("pattern_type") == "presence"
                for item in objects
            )
        )

    def test_support_camera_capture_enqueues_multimodal_camera_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            runtime = TwinrRuntime(config=config)
            camera = _FakeCamera()
            harness = _SupportHarness(runtime=runtime, camera=camera, config=config)

            images = harness._build_vision_images()
            runtime.flush_long_term_memory(timeout_s=2.0)
            objects = tuple(runtime.long_term_memory.object_store.load_objects())
            runtime.shutdown(timeout_s=2.0)

        self.assertEqual(camera.capture_calls, 1)
        self.assertEqual(len(images), 1)
        self.assertTrue(any(item.memory_id.startswith("pattern:camera_use:vision_inspection:") for item in objects))

    def test_service_can_store_repeated_print_usage_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = LongTermMemoryService.from_config(_config(temp_dir))

            for _ in range(2):
                service.enqueue_multimodal_evidence(
                    event_name="print_completed",
                    modality="printer",
                    source="realtime_print",
                    message="Printed Twinr output was delivered from the realtime loop.",
                    data={"request_source": "button", "queue": "Thermal_GP58"},
                )
            service.flush(timeout_s=2.0)
            objects = tuple(service.object_store.load_objects())
            service.shutdown()

        print_pattern = next(item for item in objects if item.memory_id.startswith("pattern:print:button:"))
        self.assertEqual(print_pattern.status, "active")
        self.assertGreaterEqual((print_pattern.attributes or {}).get("support_count", 0), 2)


if __name__ == "__main__":
    unittest.main()
