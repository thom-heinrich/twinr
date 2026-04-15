from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from threading import Lock
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.workflows.realtime_runtime.background import TwinrRealtimeBackgroundMixin
from twinr.agent.workflows.realtime_runtime.support import TwinrRealtimeSupportMixin
from twinr.agent.base_agent import TwinrConfig
from twinr.memory.longterm.core.models import (
    LongTermConsolidationResultV1,
    LongTermMemoryObjectV1,
    LongTermMultimodalEvidence,
    LongTermSourceRefV1,
)
from twinr.memory.longterm.ingestion.multimodal import LongTermMultimodalExtractor
from twinr.memory.longterm.runtime.service import LongTermMemoryService
from twinr.agent.base_agent import TwinrRuntime


class _FakeCameraCapture:
    def __init__(self) -> None:
        self.source_device = "/dev/video0"
        self.input_format = "mjpeg"
        self.data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`h\xf8\x0f"
            b"\x00\x02\x03\x01\x80$a\xf5\x97\x00\x00\x00\x00IEND\xaeB`\x82"
        )
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
        self.config = runtime.config
        self._sensor_observation_queue: Queue[tuple[dict[str, object], tuple[str, ...]]] = Queue(maxsize=1)
        self._conversation_session_active = False
        self.emit = lambda _line: None

    def _run_matching_sensor_automations(self, *, facts: dict[str, object], event_names: tuple[str, ...]) -> bool:
        return False


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


def _smart_home_sensor_facts() -> dict[str, object]:
    return {
        "pir": {
            "motion_detected": False,
            "low_motion": True,
            "no_motion_for_s": 30.0,
        },
        "camera": {
            "person_visible": False,
            "person_visible_for_s": 0.0,
            "looking_toward_device": False,
            "body_pose": "unknown",
            "smiling": False,
            "hand_or_object_near_camera": False,
            "hand_or_object_near_camera_for_s": 0.0,
        },
        "smart_home": {
            "sensor_stream_live": True,
            "motion_detected": True,
            "motion_entity_ids": ["route:192.168.178.22:motion-node-1"],
            "motion_active_by_entity": {
                "route:192.168.178.22:motion-node-1": True,
                "route:192.168.178.22:motion-node-2": False,
            },
            "device_offline": True,
            "offline_entity_ids": ["route:192.168.178.22:motion-node-2"],
            "alarm_triggered": False,
            "recent_events": [
                {
                    "event_id": "route-event:192.168.178.22:evt-1",
                    "provider": "hue",
                    "entity_id": "route:192.168.178.22:motion-node-1",
                    "event_kind": "motion_detected",
                    "observed_at": "2026-03-19T08:00:00Z",
                    "label": "Esszimmer 1",
                    "area": "Erdgeschoss",
                    "details": {"route_id": "192.168.178.22"},
                },
                {
                    "event_id": "route-event:192.168.178.22:evt-2",
                    "provider": "hue",
                    "entity_id": "route:192.168.178.22:motion-node-2",
                    "event_kind": "device_offline",
                    "observed_at": "2026-03-19T08:01:00Z",
                    "label": "Flur Sensor",
                    "area": "Erdgeschoss",
                    "details": {"route_id": "192.168.178.22"},
                },
            ],
        },
    }


def _respeaker_sensor_facts(
    *,
    observed_at: float = 8.0,
    presence_session_id: int = 17,
    speech_detected: bool = True,
    room_quiet: bool = False,
    recent_speech_age_s: float = 0.2,
    direction_confidence: float = 0.81,
    background_media_likely: bool = False,
    room_busy_or_overlapping: bool = False,
    quiet_window_open: bool = False,
    presence_audio_active: bool = True,
    recent_follow_up_speech: bool = False,
    barge_in_recent: bool = False,
    resume_window_open: bool = False,
) -> dict[str, object]:
    return {
        "sensor": {
            "inspected": False,
            "observed_at": observed_at,
            "captured_at": observed_at,
            "presence_session_id": presence_session_id,
        },
        "pir": {
            "motion_detected": False,
            "low_motion": True,
            "no_motion_for_s": 30.0,
        },
        "camera": {
            "person_visible": False,
            "person_visible_for_s": 0.0,
            "looking_toward_device": False,
            "body_pose": "unknown",
            "smiling": False,
            "hand_or_object_near_camera": False,
            "hand_or_object_near_camera_for_s": 0.0,
        },
        "vad": {
            "speech_detected": speech_detected,
            "speech_detected_for_s": 0.5 if speech_detected else 0.0,
            "quiet": not speech_detected,
            "quiet_for_s": 0.0 if speech_detected else 12.0,
            "distress_detected": False,
            "room_quiet": room_quiet,
            "recent_speech_age_s": recent_speech_age_s,
            "assistant_output_active": False,
            "signal_source": "respeaker_xvf3800",
        },
        "respeaker": {
            "runtime_mode": "audio_ready",
            "host_control_ready": True,
            "transport_reason": None,
            "azimuth_deg": 285,
            "direction_confidence": direction_confidence,
            "non_speech_audio_likely": False,
            "background_media_likely": background_media_likely,
            "speech_overlap_likely": room_busy_or_overlapping,
            "barge_in_detected": barge_in_recent,
            "mute_active": False,
            "claim_contract": {
                "speech_detected": {
                    "captured_at": observed_at,
                    "source": "respeaker_xvf3800",
                    "source_type": "observed",
                    "confidence": 0.76,
                    "sensor_window_ms": 0,
                    "memory_class": "ephemeral_state",
                    "session_id": presence_session_id,
                    "requires_confirmation": False,
                },
                "recent_speech_age_s": {
                    "captured_at": observed_at,
                    "source": "respeaker_xvf3800",
                    "source_type": "observed",
                    "confidence": 0.76,
                    "sensor_window_ms": 0,
                    "memory_class": "ephemeral_state",
                    "session_id": presence_session_id,
                    "requires_confirmation": False,
                },
                "direction_confidence": {
                    "captured_at": observed_at,
                    "source": "respeaker_xvf3800",
                    "source_type": "observed",
                    "confidence": max(0.4, direction_confidence),
                    "sensor_window_ms": 0,
                    "memory_class": "ephemeral_state",
                    "session_id": presence_session_id,
                    "requires_confirmation": False,
                },
                "azimuth_deg": {
                    "captured_at": observed_at,
                    "source": "respeaker_xvf3800",
                    "source_type": "observed",
                    "confidence": max(0.4, direction_confidence),
                    "sensor_window_ms": 0,
                    "memory_class": "ephemeral_state",
                    "session_id": presence_session_id,
                    "requires_confirmation": False,
                },
            },
        },
        "audio_policy": {
            "presence_audio_active": presence_audio_active,
            "recent_follow_up_speech": recent_follow_up_speech,
            "room_busy_or_overlapping": room_busy_or_overlapping,
            "quiet_window_open": quiet_window_open,
            "non_speech_audio_likely": False,
            "background_media_likely": background_media_likely,
            "barge_in_recent": barge_in_recent,
            "speaker_direction_stable": True,
            "mute_blocks_voice_capture": False,
            "resume_window_open": resume_window_open,
            "initiative_block_reason": None,
            "speech_delivery_defer_reason": None,
            "runtime_alert_code": "ready",
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
            harness.handle_sensor_observation(_sensor_facts(), ("pir.motion_detected", "camera.person_visible"))
            harness._maybe_run_sensor_automation()
            runtime.flush_long_term_memory(timeout_s=2.0)
            objects = tuple(runtime.long_term_memory.object_store.load_objects())
            runtime.shutdown(timeout_s=2.0)

        self.assertIn("pir", queued_facts)
        self.assertIn("pir.motion_detected", queued_events)
        self.assertIn("camera.person_visible", queued_events)
        self.assertTrue(
            any(
                item.kind == "pattern" and (item.attributes or {}).get("pattern_type") == "presence"
                for item in objects
            )
        )

    def test_background_sensor_handler_keeps_only_latest_pending_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = TwinrRuntime(config=_config(temp_dir))
            harness = _BackgroundHarness(runtime)
            first_facts = _sensor_facts()
            second_facts = _sensor_facts()
            second_facts["camera"]["person_visible_for_s"] = 42.0

            harness.handle_sensor_observation(first_facts, ("camera.person_visible",))
            harness.handle_sensor_observation(second_facts, ("camera.person_visible", "pir.motion_detected"))
            queued_facts, queued_events = harness._sensor_observation_queue.get_nowait()
            runtime.shutdown(timeout_s=2.0)

        self.assertEqual(queued_facts["camera"]["person_visible_for_s"], 42.0)
        self.assertEqual(queued_events, ("camera.person_visible", "pir.motion_detected"))

    def test_background_sensor_handler_merges_smart_home_into_latest_live_facts_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                _config(temp_dir),
                smart_home_same_room_entity_ids=("route:192.168.178.22:motion-node-1",),
                smart_home_same_room_motion_window_s=60.0,
                smart_home_stream_stale_after_s=120.0,
            )
            runtime = TwinrRuntime(
                config=config
            )
            harness = _BackgroundHarness(runtime)
            local_facts = _sensor_facts()
            local_facts["sensor"] = {"observed_at": 8.0, "captured_at": 8.0}

            harness.handle_sensor_observation(local_facts, ("camera.person_visible",))
            harness.handle_sensor_observation(
                {"smart_home": _smart_home_sensor_facts()["smart_home"]},
                ("smart_home.motion_detected", "smart_home.device_offline"),
            )
            queued_facts, queued_events = harness._sensor_observation_queue.get_nowait()
            runtime.shutdown(timeout_s=2.0)

        self.assertIn("camera", queued_facts)
        self.assertIn("smart_home", queued_facts)
        self.assertTrue(queued_facts["near_device_presence"]["occupied_likely"])
        self.assertTrue(queued_facts["room_context"]["same_room_motion_recent"])
        self.assertTrue(queued_facts["home_context"]["device_offline"])
        self.assertEqual(queued_facts["person_state"]["presence_state"]["state"], "occupied_visible")
        self.assertEqual(queued_facts["person_state"]["home_context_state"]["state"], "device_offline")
        self.assertIn("smart_home.motion_detected", queued_events)
        self.assertIn("room_context.same_room_motion_recent", queued_events)
        self.assertIn("home_context.device_offline", queued_events)

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

    def test_support_camera_capture_skips_multimodal_usage_when_persistence_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _config(temp_dir)
            runtime = TwinrRuntime(config=config)
            camera = _FakeCamera()
            harness = _SupportHarness(runtime=runtime, camera=camera, config=config)
            harness._persist_multimodal_evidence = False

            images = harness._build_vision_images()
            runtime.flush_long_term_memory(timeout_s=2.0)
            objects = tuple(runtime.long_term_memory.object_store.load_objects())
            runtime.shutdown(timeout_s=2.0)

        self.assertEqual(camera.capture_calls, 1)
        self.assertEqual(len(images), 1)
        self.assertFalse(any(item.memory_id.startswith("pattern:camera_use:vision_inspection:") for item in objects))

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

    def test_multimodal_extractor_emits_respeaker_audio_routine_seed_without_raw_audio(self) -> None:
        extractor = LongTermMultimodalExtractor()
        evidence = LongTermMultimodalEvidence(
            event_name="sensor_observation",
            modality="sensor",
            source="proactive_monitor",
            message="Changed multimodal sensor observation recorded.",
            data={
                "facts": _respeaker_sensor_facts(),
                "event_names": ["audio_policy.presence_audio_active", "vad.speech_detected"],
            },
            created_at=datetime(2026, 3, 19, 8, 0, tzinfo=timezone.utc),
        )

        extraction = extractor.extract_evidence(evidence)
        pattern = next(
            item
            for item in extraction.candidate_objects
            if item.memory_id == "pattern:audio_interaction:conversation_start_audio:morning"
        )
        attrs = dict(pattern.attributes or {})
        episode = extraction.episode
        self.assertEqual(attrs["memory_domain"], "respeaker_audio_routine")
        self.assertEqual(attrs["memory_class"], "session_memory")
        self.assertEqual(attrs["source_sensor"], "respeaker_xvf3800")
        self.assertEqual(attrs["source_type"], "observed")
        self.assertEqual(attrs["presence_session_id"], 17)
        self.assertTrue(attrs["requires_confirmation"])
        self.assertAlmostEqual(attrs["claim_confidence"], 0.785, places=3)
        self.assertEqual(attrs["claim_source"], "respeaker_xvf3800")
        self.assertEqual(attrs["claim_names"], ("speech_detected", "recent_speech_age_s", "direction_confidence", "azimuth_deg"))
        self.assertEqual(attrs["claim_contract"]["speech_detected"]["session_id"], 17)
        self.assertEqual(attrs["captured_at"], 8.0)
        self.assertNotIn("pcm_bytes", episode.details or "")
        self.assertNotIn("raw_audio", episode.details or "")

    def test_multimodal_extractor_emits_room_agnostic_smart_home_motion_and_health_seeds(self) -> None:
        extractor = LongTermMultimodalExtractor()
        evidence = LongTermMultimodalEvidence(
            event_name="sensor_observation",
            modality="sensor",
            source="proactive_monitor",
            message="Changed multimodal sensor observation recorded.",
            data={
                "facts": _smart_home_sensor_facts(),
                "event_names": ["smart_home.motion_detected", "smart_home.device_offline"],
            },
            created_at=datetime(2026, 3, 19, 8, 2, tzinfo=timezone.utc),
        )

        extraction = extractor.extract_evidence(evidence)
        motion = next(
            item
            for item in extraction.candidate_objects
            if (item.attributes or {}).get("environment_signal_type") == "motion_node_activity"
        )
        health = next(
            item
            for item in extraction.candidate_objects
            if (item.attributes or {}).get("environment_signal_type") == "node_health"
        )

        self.assertIn("pattern:smart_home_node_activity:", motion.memory_id)
        self.assertEqual((motion.attributes or {}).get("node_id"), "route:192.168.178.22:motion-node-1")
        self.assertEqual((motion.attributes or {}).get("provider_label"), "Esszimmer 1")
        self.assertEqual((motion.attributes or {}).get("route_id"), "192.168.178.22")
        self.assertEqual(motion.source.source_type, "smart_home_sensor")
        self.assertTrue(motion.source.event_ids[0].startswith("smart_home_env:20260319T090000"))

        self.assertIn("pattern:smart_home_node_health:", health.memory_id)
        self.assertEqual((health.attributes or {}).get("health_state"), "offline")
        self.assertEqual((health.attributes or {}).get("provider_area_label"), "Erdgeschoss")

    def test_low_signal_multimodal_batches_skip_optional_midterm_compilation(self) -> None:
        source_ref = LongTermSourceRefV1(
            source_type="device_event",
            event_ids=("multimodal:button",),
            modality="button",
        )
        episode = LongTermMemoryObjectV1(
            memory_id="episode:multimodal_button",
            kind="episode",
            summary="Multimodal device event recorded: button_interaction.",
            details="Structured multimodal evidence: button_interaction",
            source=source_ref,
            status="candidate",
            confidence=0.92,
            sensitivity="normal",
            slot_key="episode:multimodal:button",
            value_key="button_interaction",
        )
        pattern = LongTermMemoryObjectV1(
            memory_id="pattern:button:green:start_listening:morning",
            kind="pattern",
            summary="The green button was used to start a conversation in the morning.",
            details="Low-confidence button usage pattern derived from a physical interaction event.",
            source=source_ref,
            status="candidate",
            confidence=0.6,
            sensitivity="low",
            slot_key="pattern:button:green:start_listening:morning",
            value_key="button_used",
            attributes={"pattern_type": "interaction", "support_count": 2, "daypart": "morning"},
        )
        result = LongTermConsolidationResultV1(
            turn_id="multimodal:button",
            occurred_at=datetime(2026, 3, 9, 8, 0, tzinfo=timezone.utc),
            episodic_objects=(episode,),
            durable_objects=(pattern,),
            deferred_objects=(),
            conflicts=(),
            graph_edges=(),
        )

        self.assertFalse(LongTermMemoryService._should_include_midterm_in_multimodal_reflection(result))

    def test_richer_multimodal_objects_keep_optional_midterm_compilation(self) -> None:
        source_ref = LongTermSourceRefV1(
            source_type="device_event",
            event_ids=("multimodal:test",),
            modality="sensor",
        )
        episode = LongTermMemoryObjectV1(
            memory_id="episode:multimodal_test",
            kind="episode",
            summary="Multimodal device event recorded: custom_fact.",
            details="Structured multimodal evidence: {}",
            source=source_ref,
            status="candidate",
            confidence=0.92,
            sensitivity="normal",
            slot_key="episode:multimodal:test",
            value_key="custom_fact",
        )
        fact = LongTermMemoryObjectV1(
            memory_id="fact:hydration",
            kind="fact",
            summary="The user prefers a glass of water in the morning.",
            details="Derived from a richer multimodal signal.",
            source=source_ref,
            status="candidate",
            confidence=0.72,
            sensitivity="low",
            slot_key="fact:hydration_preference",
            value_key="water_morning",
            attributes={"fact_type": "preference"},
        )
        result = LongTermConsolidationResultV1(
            turn_id="multimodal:test",
            occurred_at=datetime(2026, 3, 9, 8, 0, tzinfo=timezone.utc),
            episodic_objects=(episode,),
            durable_objects=(fact,),
            deferred_objects=(),
            conflicts=(),
            graph_edges=(),
        )

        self.assertTrue(LongTermMemoryService._should_include_midterm_in_multimodal_reflection(result))


if __name__ == "__main__":
    unittest.main()
