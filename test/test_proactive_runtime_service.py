from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import sys
import unittest
from unittest.mock import Mock, patch


# pylint: disable=no-member

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupDecision
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime.attention_targeting import MultimodalAttentionTargetSnapshot
from twinr.proactive.runtime.display_gesture_emoji import DisplayGestureEmojiDecision
from twinr.proactive.runtime.perception_orchestrator import (
    PerceptionAttentionRuntimeSnapshot,
    PerceptionGestureRuntimeSnapshot,
    PerceptionRuntimeSnapshot,
)
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot
from twinr.proactive.runtime import service_attention_helpers
from twinr.proactive.runtime import service as proactive_service_mod
from twinr.proactive.runtime.service import (
    ProactiveCoordinator,
    build_default_proactive_monitor,
    _proactive_pcm_capture_conflicts_with_voice_orchestrator,
    _voice_orchestrator_capture_device,
)
from twinr.proactive.runtime.service_gesture_helpers import record_gesture_debug_tick
from twinr.proactive.runtime.service_impl.builder import (
    BuildDefaultProactiveMonitorDependencies,
    build_default_proactive_monitor as build_internal_proactive_monitor,
)
from twinr.proactive.social.engine import (
    SocialAudioObservation,
    SocialFineHandGesture,
    SocialObservation,
    SocialTriggerDecision,
    SocialTriggerPriority,
    SocialVisionObservation,
)
from twinr.proactive.social.observers import ProactiveAudioSnapshot
from twinr.proactive.social.observers import ProactiveVisionSnapshot
from twinr.proactive.social.perception_stream import (
    PerceptionGestureStreamObservation,
    PerceptionStreamObservation,
)


class ProactiveCoordinatorTests(unittest.TestCase):
    @staticmethod
    def _runtime_stub() -> SimpleNamespace:
        return SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )

    @staticmethod
    def _builder_dependencies() -> BuildDefaultProactiveMonitorDependencies:
        return BuildDefaultProactiveMonitorDependencies(
            social_trigger_engine_cls=proactive_service_mod.SocialTriggerEngine,
            configured_pir_monitor=proactive_service_mod.configured_pir_monitor,
            presence_session_cls=proactive_service_mod.PresenceSessionController,
            null_audio_observer_cls=proactive_service_mod.NullAudioObservationProvider,
            ambient_audio_sampler_cls=proactive_service_mod.AmbientAudioSampler,
            ambient_audio_observer_cls=proactive_service_mod.AmbientAudioObservationProvider,
            base_signal_provider_cls=proactive_service_mod.ReSpeakerSignalProvider,
            scheduled_signal_provider_cls=proactive_service_mod.ScheduledReSpeakerSignalProvider,
            respeaker_audio_observer_cls=proactive_service_mod.ReSpeakerAudioObservationProvider,
            openai_vision_provider_cls=proactive_service_mod.OpenAIVisionObservationProvider,
            aideck_vision_provider_cls=proactive_service_mod.AIDeckOpenAIVisionObservationProvider,
            local_vision_provider_cls=proactive_service_mod.LocalAICameraObservationProvider,
            vision_reviewer_cls=proactive_service_mod.OpenAIProactiveVisionReviewer,
            vision_frame_buffer_cls=proactive_service_mod.ProactiveVisionFrameBuffer,
            portrait_match_provider_cls=proactive_service_mod.PortraitMatchProvider,
            coordinator_cls=proactive_service_mod.ProactiveCoordinator,
            monitor_service_cls=proactive_service_mod.ProactiveMonitorService,
        )

    def test_shared_capture_conflict_detects_generic_voice_alias_on_same_card(self) -> None:
        config = TwinrConfig(
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            voice_orchestrator_audio_device="default",
            audio_input_device="sysdefault:CARD=Array",
            proactive_audio_enabled=True,
        )

        self.assertEqual(_voice_orchestrator_capture_device(config), "sysdefault:CARD=Array")
        self.assertTrue(_proactive_pcm_capture_conflicts_with_voice_orchestrator(config))

    def test_shared_capture_conflict_can_require_active_owner_for_standalone_checks(self) -> None:
        config = TwinrConfig(
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            voice_orchestrator_audio_device="plughw:CARD=Array,DEV=0",
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_enabled=True,
        )

        with patch("twinr.proactive.runtime.service_impl.compat.loop_lock_owner", side_effect=[None, None]):
            self.assertFalse(
                _proactive_pcm_capture_conflicts_with_voice_orchestrator(
                    config,
                    require_active_owner=True,
                )
            )

    def test_audio_policy_does_not_block_safety_triggers(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        decision = SocialTriggerDecision(
            trigger_id="possible_fall",
            prompt="prompt",
            reason="reason",
            observed_at=1.0,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
        )
        audio_policy = ReSpeakerAudioPolicySnapshot(
            observed_at=1.0,
            initiative_block_reason="room_busy_or_overlapping",
        )

        blocked_reason = ProactiveCoordinator._audio_policy_block_reason(
            service,
            decision,
            presence_snapshot=SimpleNamespace(reason="clear"),
            audio_policy_snapshot=audio_policy,
        )

        self.assertIsNone(blocked_reason)

    def test_audio_policy_blocks_non_safety_trigger_when_reason_exists(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        decision = SocialTriggerDecision(
            trigger_id="attention_window",
            prompt="prompt",
            reason="reason",
            observed_at=1.0,
            priority=SocialTriggerPriority.ATTENTION_WINDOW,
        )
        audio_policy = ReSpeakerAudioPolicySnapshot(
            observed_at=1.0,
            initiative_block_reason="room_busy_or_overlapping",
        )

        blocked_reason = ProactiveCoordinator._audio_policy_block_reason(
            service,
            decision,
            presence_snapshot=SimpleNamespace(reason="clear"),
            audio_policy_snapshot=audio_policy,
        )

        self.assertEqual(blocked_reason, "room_busy_or_overlapping")

    def test_dispatch_gesture_wakeup_with_fresh_context_primes_sensor_context_before_handler(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        events: list[tuple[str, object]] = []

        def _prime(**kwargs) -> None:
            events.append(("prime", kwargs["observed_at"]))

        def _handle(decision: GestureWakeupDecision) -> bool:
            events.append(("handle", decision.reason))
            return True

        service._prime_gesture_wakeup_sensor_context = _prime
        service._handle_gesture_wakeup_decision = _handle

        handled = ProactiveCoordinator._dispatch_gesture_wakeup_with_fresh_context(
            service,
            observed_at=10.0,
            vision_snapshot=SimpleNamespace(
                observation=SocialVisionObservation(
                    person_visible=True,
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                )
            ),
            decision=GestureWakeupDecision(
                active=True,
                reason="gesture_wakeup:peace_sign",
                confidence=0.91,
            ),
        )

        self.assertTrue(handled)
        self.assertEqual(
            events,
            [
                ("prime", 10.0),
                ("handle", "gesture_wakeup:peace_sign"),
            ],
        )

    def test_prime_gesture_wakeup_sensor_context_exports_current_snapshot_facts(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        exported: list[tuple[dict[str, object], tuple[str, ...]]] = []
        camera_update = SimpleNamespace(
            snapshot=SimpleNamespace(person_visible=True),
            event_names=("camera.fine_hand_gesture_detected",),
        )
        audio_snapshot = ProactiveAudioSnapshot(
            observation=SocialAudioObservation(
                speech_detected=False,
                recent_speech_age_s=9.0,
            )
        )

        service.observation_handler = lambda facts, event_names: exported.append((facts, event_names))
        service._observe_audio_for_attention_refresh = lambda now: audio_snapshot
        service._observe_audio_policy = lambda **kwargs: SimpleNamespace()
        service._observe_camera_surface = lambda observation, inspected: camera_update
        service._observe_presence = lambda **kwargs: SimpleNamespace()
        service._build_automation_facts = lambda observation, **kwargs: {
            "camera": {"person_visible": observation.vision.person_visible},
            "person_state": {
                "interaction_ready": True,
                "targeted_inference_blocked": False,
                "recommended_channel": "speech",
            },
        }
        service._derive_sensor_events = lambda facts, camera_event_names=(): (
            *camera_event_names,
            "person_state.interaction_ready",
        )

        ProactiveCoordinator._prime_gesture_wakeup_sensor_context(
            service,
            observed_at=11.0,
            vision_snapshot=SimpleNamespace(
                observation=SocialVisionObservation(
                    person_visible=True,
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                )
            ),
        )

        self.assertEqual(len(exported), 1)
        facts, event_names = exported[0]
        self.assertEqual(facts["camera"]["person_visible"], True)
        self.assertEqual(facts["person_state"]["recommended_channel"], "speech")
        self.assertEqual(
            event_names,
            ("camera.fine_hand_gesture_detected", "person_state.interaction_ready"),
        )

    def test_run_busy_audio_only_uses_non_blocking_audio_refresh_path(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        calls: list[str] = []
        audio_snapshot = ProactiveAudioSnapshot(
            observation=SocialAudioObservation(
                speech_detected=False,
                assistant_output_active=False,
            )
        )

        service._note_audio_observer_runtime_context = lambda **_kwargs: calls.append("note_context")
        service._observe_audio_for_busy_runtime = lambda **_kwargs: calls.append("busy_refresh") or audio_snapshot
        service._observe_audio_safe = lambda: calls.append("observe_audio_safe") or audio_snapshot
        service._observe_audio_policy = lambda **_kwargs: SimpleNamespace()
        service._is_low_motion = lambda now, motion_active: False
        service._record_observation_if_changed = (
            lambda observation, **_kwargs: calls.append(f"record:{observation.audio.speech_detected}")
        )

        result = ProactiveCoordinator._run_busy_audio_only(
            service,
            now=15.0,
            motion_active=False,
            runtime_status_value="processing",
        )

        self.assertIsInstance(result, proactive_service_mod.ProactiveTickResult)
        self.assertEqual(calls, ["note_context", "busy_refresh", "record:False"])

    def test_update_display_attention_follow_consumes_perception_orchestrator(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        published: list[dict[str, object]] = []
        speaker_association = ReSpeakerSpeakerAssociationSnapshot(
            observed_at=12.0,
            state="primary_visible_person_associated",
            associated=True,
            target_id="primary_visible_person",
            confidence=0.92,
            camera_person_count=1,
            direction_confidence=0.89,
            azimuth_deg=15,
            primary_person_zone="right",
        )
        attention_target = MultimodalAttentionTargetSnapshot(
            observed_at=12.0,
            state="active_visible_speaker",
            active=True,
            target_horizontal="right",
            target_zone="right",
            target_center_x=0.72,
            focus_source="speaker_association",
            speaker_locked=True,
            confidence=0.88,
        )
        perception_snapshot = PerceptionRuntimeSnapshot(
            observed_at=12.0,
            source="display_attention_refresh",
            captured_at=11.8,
            attention=PerceptionAttentionRuntimeSnapshot(
                live_facts={
                    "camera": {"person_visible": True, "primary_person_center_x": 0.72},
                    "vad": {"speech_detected": True},
                    "respeaker": {"azimuth_deg": 15, "direction_confidence": 0.89},
                    "audio_policy": {"speaker_direction_stable": True},
                    "speaker_association": speaker_association.to_automation_facts(),
                    "attention_target": attention_target.to_automation_facts(),
                },
                speaker_association=speaker_association,
                attention_target=attention_target,
                attention_target_debug={"source": "test"},
            ),
        )
        orchestrator_calls: list[dict[str, object]] = []
        service.latest_presence_snapshot = SimpleNamespace(session_id=7)
        service.latest_identity_fusion_snapshot = None
        service.runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
        service.config = TwinrConfig(project_root=".")
        service.attention_servo_controller = None
        service.perception_orchestrator = SimpleNamespace(
            observe_attention=lambda **kwargs: orchestrator_calls.append(kwargs) or perception_snapshot
        )
        service.display_attention_publisher = SimpleNamespace(
            publish_from_facts=lambda config, live_facts: (
                published.append({"config": config, "live_facts": live_facts}) or SimpleNamespace(action="published")
            )
        )
        service._record_fault = lambda **_kwargs: None
        service._append_ops_event = lambda **_kwargs: None
        service._emit = lambda _line: None
        service._last_attention_follow_pipeline_key = None
        service._last_attention_servo_follow_key = None
        service.latest_perception_runtime_snapshot = None
        service.latest_speaker_association_snapshot = None
        service.latest_attention_target_snapshot = None

        with (
            patch("twinr.proactive.runtime.service_attention_helpers.record_attention_follow_pipeline_if_changed"),
            patch("twinr.proactive.runtime.service_attention_helpers.update_attention_servo_follow"),
        ):
            result = ProactiveCoordinator._update_display_attention_follow(
                service,
                source="display_attention_refresh",
                observed_at=12.0,
                camera_snapshot=SimpleNamespace(
                    last_camera_frame_at=11.8,
                    to_automation_facts=lambda: {"person_visible": True, "primary_person_center_x": 0.72},
                ),
                audio_observation=SocialAudioObservation(
                    speech_detected=True,
                    azimuth_deg=15,
                    direction_confidence=0.89,
                ),
                audio_policy_snapshot=SimpleNamespace(speaker_direction_stable=True),
            )

        self.assertEqual(len(orchestrator_calls), 1)
        self.assertIs(service.latest_perception_runtime_snapshot, perception_snapshot)
        self.assertEqual(service.latest_speaker_association_snapshot, speaker_association)
        self.assertEqual(service.latest_attention_target_snapshot, attention_target)
        self.assertEqual(len(published), 1)
        self.assertEqual(
            published[0]["live_facts"]["attention_target"],
            attention_target.to_automation_facts(),
        )
        self.assertEqual(result.action, "published")

    def test_update_display_attention_follow_skips_face_publish_on_hdmi_wayland(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        speaker_association = ReSpeakerSpeakerAssociationSnapshot(
            observed_at=12.0,
            state="primary_visible_person_associated",
            associated=True,
            target_id="primary_visible_person",
            confidence=0.92,
            camera_person_count=1,
            direction_confidence=0.89,
            azimuth_deg=15,
            primary_person_zone="right",
        )
        attention_target = MultimodalAttentionTargetSnapshot(
            observed_at=12.0,
            state="active_visible_speaker",
            active=True,
            target_horizontal="right",
            target_zone="right",
            target_center_x=0.72,
            focus_source="speaker_association",
            speaker_locked=True,
            confidence=0.88,
        )
        perception_snapshot = PerceptionRuntimeSnapshot(
            observed_at=12.0,
            source="display_attention_refresh",
            captured_at=11.8,
            attention=PerceptionAttentionRuntimeSnapshot(
                live_facts={
                    "camera": {"person_visible": True, "primary_person_center_x": 0.72},
                    "vad": {"speech_detected": True},
                    "speaker_association": speaker_association.to_automation_facts(),
                    "attention_target": attention_target.to_automation_facts(),
                },
                speaker_association=speaker_association,
                attention_target=attention_target,
                attention_target_debug={"source": "test"},
            ),
        )
        published: list[dict[str, object]] = []
        service.latest_presence_snapshot = SimpleNamespace(session_id=7)
        service.latest_identity_fusion_snapshot = None
        service.runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
        service.config = TwinrConfig(project_root=".", display_driver="hdmi_wayland")
        service.attention_servo_controller = None
        service.perception_orchestrator = SimpleNamespace(
            observe_attention=lambda **_kwargs: perception_snapshot
        )
        service.display_attention_publisher = SimpleNamespace(
            publish_from_facts=lambda **kwargs: published.append(kwargs) or SimpleNamespace(action="published")
        )
        service._record_fault = lambda **_kwargs: None
        service._append_ops_event = lambda **_kwargs: None
        service._emit = lambda _line: None
        service._last_attention_follow_pipeline_key = None
        service._last_attention_servo_follow_key = None
        service.latest_perception_runtime_snapshot = None
        service.latest_speaker_association_snapshot = None
        service.latest_attention_target_snapshot = None

        with (
            patch("twinr.proactive.runtime.service_attention_helpers.record_attention_follow_pipeline_if_changed"),
            patch("twinr.proactive.runtime.service_attention_helpers.update_attention_servo_follow"),
        ):
            result = ProactiveCoordinator._update_display_attention_follow(
                service,
                source="display_attention_refresh",
                observed_at=12.0,
                camera_snapshot=SimpleNamespace(
                    last_camera_frame_at=11.8,
                    to_automation_facts=lambda: {"person_visible": True, "primary_person_center_x": 0.72},
                ),
                audio_observation=SocialAudioObservation(
                    speech_detected=True,
                    azimuth_deg=15,
                    direction_confidence=0.89,
                ),
                audio_policy_snapshot=SimpleNamespace(speaker_direction_stable=True),
            )

        self.assertEqual(service.latest_speaker_association_snapshot, speaker_association)
        self.assertEqual(service.latest_attention_target_snapshot, attention_target)
        self.assertEqual(published, [])
        self.assertIsNone(result)

    def test_non_authoritative_offline_automation_tick_does_not_release_servo(self) -> None:
        controller = SimpleNamespace(update=Mock())
        coordinator = SimpleNamespace(
            attention_servo_controller=controller,
            config=TwinrConfig(project_root="."),
            vision_observer=SimpleNamespace(supports_attention_refresh=True),
        )
        camera_snapshot = SimpleNamespace(
            camera_online=False,
            camera_ready=False,
            camera_ai_ready=False,
            camera_error=None,
            last_camera_frame_at=None,
            person_visible=False,
            primary_person_box=None,
        )

        with (
            patch(
                "twinr.proactive.runtime.service_attention_helpers.display_attention_refresh_supported",
                return_value=True,
            ),
            patch.object(
                service_attention_helpers,
                "record_attention_servo_follow_if_changed",
            ) as record_follow,
            patch.object(
                service_attention_helpers,
                "record_attention_servo_forensic_tick",
            ) as record_forensic,
        ):
            service_attention_helpers.update_attention_servo_follow(
                coordinator,
                source="automation_observation",
                observed_at=10.0,
                camera_snapshot=camera_snapshot,
                attention_target=None,
            )

        controller.update.assert_not_called()
        record_follow.assert_called_once()
        self.assertEqual(record_follow.call_args.kwargs["decision"].reason, "ignored_non_authoritative_source")
        record_forensic.assert_called_once()

    def test_authoritative_offline_display_tick_still_fail_closes_servo(self) -> None:
        controller = SimpleNamespace(update=Mock())
        coordinator = SimpleNamespace(
            attention_servo_controller=controller,
            config=TwinrConfig(project_root="."),
            vision_observer=SimpleNamespace(supports_attention_refresh=True),
        )
        camera_snapshot = SimpleNamespace(
            camera_online=False,
            camera_ready=False,
            camera_ai_ready=False,
            camera_error=None,
            last_camera_frame_at=None,
            person_visible=False,
            primary_person_box=None,
        )

        with (
            patch(
                "twinr.proactive.runtime.service_attention_helpers.display_attention_refresh_supported",
                return_value=True,
            ),
            patch.object(
                service_attention_helpers,
                "record_attention_servo_follow_if_changed",
            ) as record_follow,
            patch.object(
                service_attention_helpers,
                "record_attention_servo_forensic_tick",
            ) as record_forensic,
        ):
            service_attention_helpers.update_attention_servo_follow(
                coordinator,
                source="display_attention_refresh",
                observed_at=10.0,
                camera_snapshot=camera_snapshot,
                attention_target=None,
            )

        controller.update.assert_not_called()
        record_follow.assert_called_once()
        self.assertEqual(record_follow.call_args.kwargs["decision"].reason, "camera_offline")
        record_forensic.assert_called_once()

    def test_display_gesture_refresh_consumes_perception_orchestrator(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        calls: list[tuple[str, object]] = []

        class _NullContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        snapshot = ProactiveVisionSnapshot(
            observation=SocialVisionObservation(
                hand_or_object_near_camera=True,
                fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
            ),
            response_text="gesture",
            captured_at=13.5,
        )
        gesture_runtime = PerceptionRuntimeSnapshot(
            observed_at=14.0,
            source="gesture_fast",
            captured_at=13.5,
            gesture=PerceptionGestureRuntimeSnapshot(
                ack_decision=DisplayGestureEmojiDecision(
                    active=True,
                    reason="fine_hand:peace_sign",
                    hold_seconds=0.4,
                ),
                wakeup_decision=GestureWakeupDecision(
                    active=True,
                    reason="gesture_wakeup:peace_sign",
                    confidence=0.91,
                ),
            ),
        )
        service.display_gesture_emoji_publisher = SimpleNamespace()
        service.vision_observer = SimpleNamespace(supports_gesture_refresh=True)
        service.config = TwinrConfig(
            project_root=".",
            display_driver="hdmi_wayland",
            display_gesture_refresh_interval_s=0.2,
        )
        service.runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
        service.clock = lambda: 14.0
        service._last_display_gesture_refresh_at = None
        service._last_gesture_vision_refresh_mode = "gesture_fast"
        service._gesture_forensics = SimpleNamespace(bind_refresh=lambda **_kwargs: _NullContext())
        service.perception_orchestrator = SimpleNamespace(
            observe_gesture=lambda **kwargs: calls.append(("orchestrator", kwargs)) or gesture_runtime
        )
        service._observe_vision_for_gesture_refresh = lambda: snapshot
        service._record_vision_snapshot_safe = lambda current: calls.append(("record_snapshot", current))
        service._remember_display_attention_camera_semantics = (
            lambda **kwargs: calls.append(("remember", kwargs["source"]))
        )
        service._gesture_observation_trace_details = lambda observation: {
            "fine_hand_gesture": observation.fine_hand_gesture.value
        }
        service._trace_gesture_ack_lane_decision = lambda **kwargs: calls.append(("trace_ack", kwargs["decision"]))
        service._trace_gesture_wakeup_lane_decision = lambda **kwargs: calls.append(("trace_wakeup", kwargs["decision"]))
        service._publish_display_gesture_decision = lambda decision: calls.append(("publish", decision)) or SimpleNamespace(
            action="published"
        )
        service._dispatch_gesture_wakeup_with_fresh_context = (
            lambda **kwargs: calls.append(("wakeup", kwargs["decision"])) or True
        )
        service._trace_gesture_publish_decision = lambda **kwargs: calls.append(("trace_publish", kwargs["decision"]))
        service._record_gesture_debug_tick = lambda **kwargs: calls.append(("debug", kwargs["outcome"]))
        service._record_fault = lambda **kwargs: calls.append(("fault", kwargs["event"]))
        service.latest_presence_snapshot = None
        service.latest_perception_runtime_snapshot = None

        refreshed = ProactiveCoordinator.refresh_display_gesture_emoji(service)

        self.assertTrue(refreshed)
        self.assertIs(service.latest_perception_runtime_snapshot, gesture_runtime)
        self.assertTrue(any(kind == "orchestrator" for kind, _payload in calls))
        self.assertTrue(any(kind == "publish" for kind, _payload in calls))
        self.assertTrue(any(kind == "wakeup" for kind, _payload in calls))
        self.assertFalse(any(kind == "fault" for kind, _payload in calls))

    def test_display_gesture_refresh_fails_closed_in_voice_runtime(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        calls: list[tuple[str, object]] = []

        service.display_gesture_emoji_publisher = SimpleNamespace()
        service.vision_observer = SimpleNamespace(supports_gesture_refresh=True)
        service.config = TwinrConfig(
            project_root=".",
            display_driver="hdmi_wayland",
            display_gesture_refresh_interval_s=0.2,
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://example.invalid/ws/orchestrator/voice",
        )
        service.runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
        service.clock = lambda: 14.0
        service._last_display_gesture_refresh_at = None
        service._record_gesture_debug_tick = lambda **kwargs: calls.append(("debug", kwargs["outcome"]))
        service._record_fault = lambda **kwargs: calls.append(("fault", kwargs["event"]))

        refreshed = ProactiveCoordinator.refresh_display_gesture_emoji(service)

        self.assertFalse(refreshed)
        self.assertIn(("debug", "unsupported"), calls)
        self.assertFalse(any(kind == "fault" for kind, _payload in calls))

    def test_display_refresh_cycle_reuses_one_combined_perception_snapshot(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        calls: list[str] = []
        shared_snapshot = ProactiveVisionSnapshot(
            observation=SocialVisionObservation(person_visible=True),
            response_text="shared",
        )

        class _VisionObserver:
            def observe_perception_stream(self):
                del self
                calls.append("combined")
                return shared_snapshot

        service.vision_observer = _VisionObserver()
        service._display_perception_cycle = None
        service._last_attention_vision_refresh_mode = None
        service._last_gesture_vision_refresh_mode = None
        service._observe_vision_with_method = lambda observe_fn: observe_fn()

        ProactiveCoordinator._open_display_perception_cycle(
            service,
            attention_due=True,
            gesture_due=True,
        )
        attention_snapshot = ProactiveCoordinator._observe_vision_for_attention_refresh(service)
        gesture_snapshot = ProactiveCoordinator._observe_vision_for_gesture_refresh(service)
        ProactiveCoordinator._close_display_perception_cycle(service)

        self.assertIs(attention_snapshot, shared_snapshot)
        self.assertIs(gesture_snapshot, shared_snapshot)
        self.assertEqual(calls, ["combined"])
        self.assertEqual(service._last_attention_vision_refresh_mode, "perception_stream_shared")
        self.assertEqual(service._last_gesture_vision_refresh_mode, "perception_stream_shared")

    def test_display_refresh_cycle_keeps_gesture_lane_dedicated_when_fast_refresh_exists(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        calls: list[str] = []
        shared_snapshot = ProactiveVisionSnapshot(
            observation=SocialVisionObservation(person_visible=True),
            response_text="shared",
        )
        gesture_snapshot = ProactiveVisionSnapshot(
            observation=SocialVisionObservation(
                hand_or_object_near_camera=True,
                fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
            ),
            response_text="gesture_fast",
        )

        class _VisionObserver:
            supports_gesture_refresh = True

            def observe_perception_stream(self):
                del self
                calls.append("combined")
                return shared_snapshot

            def observe_gesture(self):
                del self
                calls.append("gesture_fast")
                return gesture_snapshot

        service.vision_observer = _VisionObserver()
        service._display_perception_cycle = None
        service._last_attention_vision_refresh_mode = None
        service._last_gesture_vision_refresh_mode = None
        service._observe_vision_with_method = lambda observe_fn: observe_fn()

        ProactiveCoordinator._open_display_perception_cycle(
            service,
            attention_due=True,
            gesture_due=True,
        )
        attention_snapshot = ProactiveCoordinator._observe_vision_for_attention_refresh(service)
        gesture_observation_snapshot = ProactiveCoordinator._observe_vision_for_gesture_refresh(service)
        ProactiveCoordinator._close_display_perception_cycle(service)

        self.assertIs(attention_snapshot, shared_snapshot)
        self.assertIs(gesture_observation_snapshot, gesture_snapshot)
        self.assertEqual(calls, ["combined", "gesture_fast"])
        self.assertEqual(service._last_attention_vision_refresh_mode, "perception_stream_shared")
        self.assertEqual(service._last_gesture_vision_refresh_mode, "gesture_fast")

    def test_record_gesture_debug_tick_keeps_debug_stream_when_observation_has_perception_stream(self) -> None:
        appended: list[dict[str, object]] = []
        faults: list[tuple[str, str]] = []

        class _DebugStream:
            def append_tick(self, *, outcome, observed_at, data) -> None:
                appended.append(
                    {
                        "outcome": outcome,
                        "observed_at": observed_at,
                        "data": data,
                    }
                )

        coordinator = SimpleNamespace(
            display_gesture_debug_stream=_DebugStream(),
            _last_gesture_vision_refresh_mode="perception_stream_shared",
            vision_observer=SimpleNamespace(gesture_debug_details=lambda: {"stream_mode": "gesture_stream"}),
            _record_fault=lambda **kwargs: faults.append((kwargs["event"], str(kwargs["error"]))),
        )
        observation = SocialVisionObservation(
            camera_online=True,
            camera_ready=True,
            camera_ai_ready=True,
            hand_or_object_near_camera=True,
            fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
            perception_stream=PerceptionStreamObservation(
                gesture=PerceptionGestureStreamObservation(
                    authoritative=True,
                    activation_key="fine:peace_sign",
                    activation_token=7,
                    activation_rising=True,
                )
            ),
        )

        record_gesture_debug_tick(
            coordinator,
            observed_at=12.0,
            outcome="published",
            runtime_status_value="waiting",
            stage_ms={"observe": 1.2},
            observation=observation,
        )

        self.assertEqual(faults, [])
        self.assertEqual(len(appended), 1)
        self.assertEqual(appended[0]["outcome"], "published")
        self.assertEqual(appended[0]["observed_at"], 12.0)
        payload = cast(dict[str, object], appended[0]["data"])
        self.assertEqual(payload["gesture_stream_activation_key"], "fine:peace_sign")
        self.assertEqual(payload["gesture_stream_activation_token"], 7)
        self.assertEqual(payload["gesture_stream_activation_rising"], True)

    def test_derive_sensor_events_emits_rising_edges_only_once(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        service._last_sensor_flags = {}
        facts = {
            "pir": {"motion_detected": True},
            "vad": {"speech_detected": False},
            "audio_policy": {
                "presence_audio_active": False,
                "quiet_window_open": True,
                "resume_window_open": False,
                "room_busy_or_overlapping": False,
                "barge_in_recent": False,
            },
            "speaker_association": {"associated": True},
            "multimodal_initiative": {"ready": False},
            "ambiguous_room_guard": {"guard_active": False},
            "identity_fusion": {"matches_main_user": False},
            "portrait_match": {"matches_reference_user": False},
            "known_user_hint": {"matches_main_user": False},
            "affect_proxy": {"state": "calm"},
            "attention_target": {"session_focus_active": True},
            "person_state": {
                "interaction_ready": True,
                "safety_concern_active": False,
                "calm_personalization_allowed": False,
            },
        }

        first = ProactiveCoordinator._derive_sensor_events(
            service,
            facts,
            camera_event_names=("camera.fine_hand_gesture_detected",),
        )
        second = ProactiveCoordinator._derive_sensor_events(service, facts)

        self.assertEqual(
            first,
            (
                "camera.fine_hand_gesture_detected",
                "pir.motion_detected",
                "audio_policy.quiet_window_open",
                "speaker_association.associated",
                "attention_target.session_focus_active",
                "person_state.interaction_ready",
            ),
        )
        self.assertEqual(second, ())

    def test_dispatch_automation_observation_updates_display_and_exports_events(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        calls: list[tuple[str, object]] = []
        exported: list[tuple[dict[str, object], tuple[str, ...]]] = []
        camera_update = SimpleNamespace(
            snapshot=SimpleNamespace(person_visible=True),
            event_names=("camera.fine_hand_gesture_detected",),
        )
        publish_result = SimpleNamespace(action="published")
        service.config = TwinrConfig(display_driver="waveshare_4in2_v2")
        service.vision_observer = SimpleNamespace(supports_gesture_refresh=False)
        service.runtime = SimpleNamespace(status=SimpleNamespace(value="waiting"))
        service.observation_handler = lambda facts, event_names: exported.append((facts, event_names))
        service._remember_display_attention_camera_semantics = (
            lambda **kwargs: calls.append(("remember", kwargs["source"]))
        )
        service._observe_camera_surface = lambda observation, inspected: camera_update
        service._update_display_debug_signals = (
            lambda update, detected_trigger_ids=(): calls.append(("debug", detected_trigger_ids))
        )
        service._update_display_gesture_emoji_ack = lambda update: calls.append(("emoji", update))
        service._update_display_attention_follow = lambda **kwargs: publish_result
        service._record_display_attention_follow_if_changed = (
            lambda **kwargs: calls.append(("record_display", kwargs["publish_result"]))
        )
        service._build_automation_facts = lambda observation, **kwargs: {
            "camera": {"person_visible": observation.vision.person_visible},
            "person_state": {"interaction_ready": True},
        }
        service._derive_sensor_events = lambda facts, camera_event_names=(): (
            *camera_event_names,
            "person_state.interaction_ready",
        )
        service._record_fault = lambda **kwargs: calls.append(("fault", kwargs["event"]))

        ProactiveCoordinator._dispatch_automation_observation(
            service,
            SocialObservation(
                observed_at=12.0,
                vision=SocialVisionObservation(person_visible=True),
                audio=SocialAudioObservation(speech_detected=False),
            ),
            inspected=True,
            audio_snapshot=None,
            audio_policy_snapshot=None,
            presence_snapshot=SimpleNamespace(),
            detected_trigger_id="attention_window",
        )

        self.assertEqual(
            exported,
            [
                (
                    {
                        "camera": {"person_visible": True},
                        "person_state": {"interaction_ready": True},
                    },
                    ("camera.fine_hand_gesture_detected", "person_state.interaction_ready"),
                )
            ],
        )
        self.assertIn(("remember", "full"), calls)
        self.assertIn(("debug", ("attention_window",)), calls)
        self.assertTrue(any(kind == "emoji" for kind, _payload in calls))
        self.assertTrue(any(kind == "record_display" for kind, _payload in calls))
        self.assertFalse(any(kind == "fault" for kind, _payload in calls))

    def test_build_default_monitor_uses_local_camera_provider_by_default(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="local",
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )
        sentinel_provider = object()

        with (
            patch(
                "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
                return_value=sentinel_provider,
            ) as local_factory,
            patch(
                "twinr.proactive.runtime.service.OpenAIVisionObservationProvider",
            ) as openai_provider,
        ):
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=cast(Any, object()),
                camera=cast(Any, object()),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: False,
            )

        self.assertIsNotNone(monitor)
        assert monitor is not None
        self.assertIs(monitor.coordinator.vision_observer, sentinel_provider)
        local_factory.assert_called_once_with(config)
        openai_provider.assert_not_called()

    def test_build_default_monitor_keeps_hdmi_fbdev_attention_when_voice_orchestrator_enabled(
        self,
    ) -> None:
        config = TwinrConfig(
            display_driver="hdmi_fbdev",
            proactive_vision_provider="local",
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            display_attention_refresh_interval_s=0.6,
            display_gesture_refresh_interval_s=0.0,
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )
        sentinel_provider = object()

        with (
            patch(
                "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
                return_value=sentinel_provider,
            ) as local_factory,
            patch(
                "twinr.proactive.runtime.service.OpenAIVisionObservationProvider",
            ) as openai_provider,
        ):
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=cast(Any, object()),
                camera=cast(Any, object()),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: False,
            )

        self.assertIsNotNone(monitor)
        assert monitor is not None
        self.assertIs(monitor.coordinator.vision_observer, sentinel_provider)
        local_factory.assert_called_once_with(config)
        openai_provider.assert_not_called()

    def test_build_default_monitor_skips_hdmi_wayland_attention_when_voice_orchestrator_enabled(
        self,
    ) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="local",
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            display_attention_refresh_interval_s=0.6,
            display_gesture_refresh_interval_s=0.0,
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )

        with patch(
            "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
            return_value=object(),
        ) as local_factory:
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=cast(Any, object()),
                camera=cast(Any, object()),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: False,
            )

        self.assertIsNone(monitor)
        local_factory.assert_not_called()

    def test_build_default_monitor_skips_voice_runtime_without_hdmi_or_proactive_vision(
        self,
    ) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="local",
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8797/ws/orchestrator/voice",
            display_attention_refresh_interval_s=0.0,
            display_gesture_refresh_interval_s=0.0,
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )

        with patch(
            "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
        ) as local_factory:
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=cast(Any, object()),
                camera=cast(Any, object()),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: False,
            )

        self.assertIsNone(monitor)
        local_factory.assert_not_called()

    def test_build_default_monitor_uses_aideck_openai_provider_when_configured(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="aideck_openai",
            camera_device="aideck://192.168.4.1:5000",
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )
        sentinel_provider = object()

        with (
            patch(
                "twinr.proactive.runtime.service.AIDeckOpenAIVisionObservationProvider.from_config",
                return_value=sentinel_provider,
            ) as aideck_factory,
            patch(
                "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
            ) as local_factory,
            patch(
                "twinr.proactive.runtime.service.OpenAIVisionObservationProvider",
            ) as openai_provider,
        ):
            monitor = build_default_proactive_monitor(
                config=config,
                runtime=runtime,
                backend=cast(Any, object()),
                camera=cast(Any, object()),
                camera_lock=None,
                audio_lock=None,
                trigger_handler=lambda _decision: False,
            )

        self.assertIsNotNone(monitor)
        assert monitor is not None
        self.assertIs(monitor.coordinator.vision_observer, sentinel_provider)
        aideck_factory.assert_called_once()
        local_factory.assert_not_called()
        openai_provider.assert_not_called()

    def test_build_default_monitor_raises_when_configured_local_provider_fails(self) -> None:
        recorded_ops_events: list[dict[str, object]] = []
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="local",
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **kwargs: recorded_ops_events.append(kwargs)),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )

        with (
            patch(
                "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
                side_effect=RuntimeError("camera init exploded"),
            ) as local_factory,
            patch(
                "twinr.proactive.runtime.service.OpenAIVisionObservationProvider",
            ) as openai_provider,
        ):
            with self.assertRaisesRegex(RuntimeError, "failed to initialize"):
                build_default_proactive_monitor(
                    config=config,
                    runtime=runtime,
                    backend=cast(Any, object()),
                    camera=cast(Any, object()),
                    camera_lock=None,
                    audio_lock=None,
                    trigger_handler=lambda _decision: False,
                )

        local_factory.assert_called_once_with(config)
        openai_provider.assert_not_called()
        self.assertTrue(
            any(
                event.get("event") == "proactive_component_blocked"
                and event.get("data", {}).get("reason") == "vision_observer_init_failed"
                for event in recorded_ops_events
            )
        )

    def test_review_trigger_blocks_when_buffered_review_fails(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        recorded_faults: list[str] = []
        skipped_triggers: list[str] = []
        ops_events: list[str] = []

        service.vision_reviewer = SimpleNamespace(
            review=Mock(side_effect=RuntimeError("review exploded"))
        )
        service._record_fault = lambda **kwargs: recorded_faults.append(kwargs["event"])
        service._append_ops_event = lambda **kwargs: ops_events.append(kwargs["event"])
        service._record_trigger_skipped_vision_review_unavailable = (
            lambda decision: skipped_triggers.append(decision.trigger_id)
        )
        service._record_vision_review = lambda *_args, **_kwargs: None
        service._record_trigger_skipped_vision_review = lambda *_args, **_kwargs: None

        decision = SocialTriggerDecision(
            trigger_id="possible_fall",
            prompt="prompt",
            reason="reason",
            observed_at=1.0,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
        )

        reviewed_decision, review = ProactiveCoordinator._review_trigger(
            service,
            decision,
            observation=cast(Any, object()),
        )

        self.assertIsNone(reviewed_decision)
        self.assertIsNone(review)
        self.assertEqual(recorded_faults, ["proactive_vision_review_failed"])
        self.assertEqual(skipped_triggers, ["possible_fall"])
        self.assertEqual(ops_events, [])

    def test_review_trigger_blocks_when_buffered_review_is_unavailable(self) -> None:
        service = object.__new__(ProactiveCoordinator)
        skipped_triggers: list[str] = []
        ops_events: list[str] = []

        service.vision_reviewer = SimpleNamespace(review=Mock(return_value=None))
        service._record_fault = lambda **_kwargs: None
        service._append_ops_event = lambda **kwargs: ops_events.append(kwargs["event"])
        service._record_trigger_skipped_vision_review_unavailable = (
            lambda decision: skipped_triggers.append(decision.trigger_id)
        )
        service._record_vision_review = lambda *_args, **_kwargs: None
        service._record_trigger_skipped_vision_review = lambda *_args, **_kwargs: None

        decision = SocialTriggerDecision(
            trigger_id="possible_fall",
            prompt="prompt",
            reason="reason",
            observed_at=1.0,
            priority=SocialTriggerPriority.POSSIBLE_FALL,
        )

        reviewed_decision, review = ProactiveCoordinator._review_trigger(
            service,
            decision,
            observation=cast(Any, object()),
        )

        self.assertIsNone(reviewed_decision)
        self.assertIsNone(review)
        self.assertEqual(skipped_triggers, ["possible_fall"])
        self.assertEqual(ops_events, ["proactive_vision_review_unavailable"])

    def test_wrapper_builder_matches_internal_builder_for_provider_variants(self) -> None:
        cases = [
            (
                "local",
                {
                    "display_driver": "hdmi_wayland",
                    "proactive_vision_provider": "local",
                },
                "twinr.proactive.runtime.service.LocalAICameraObservationProvider.from_config",
            ),
            (
                "aideck_openai",
                {
                    "display_driver": "hdmi_wayland",
                    "proactive_vision_provider": "aideck_openai",
                    "camera_device": "aideck://192.168.4.1:5000",
                },
                "twinr.proactive.runtime.service.AIDeckOpenAIVisionObservationProvider.from_config",
            ),
        ]

        for provider_name, config_kwargs, patch_target in cases:
            with self.subTest(provider_name=provider_name):
                config = TwinrConfig(**config_kwargs)
                sentinel_provider = object()

                with patch(
                    patch_target,
                    return_value=sentinel_provider,
                ) as provider_factory:
                    legacy_monitor = proactive_service_mod.build_default_proactive_monitor(
                        config=config,
                        runtime=self._runtime_stub(),
                        backend=cast(Any, object()),
                        camera=cast(Any, object()),
                        camera_lock=None,
                        audio_lock=None,
                        trigger_handler=lambda _decision: False,
                    )
                    internal_monitor = build_internal_proactive_monitor(
                        config=config,
                        runtime=self._runtime_stub(),
                        backend=cast(Any, object()),
                        camera=cast(Any, object()),
                        camera_lock=None,
                        audio_lock=None,
                        trigger_handler=lambda _decision: False,
                        dependencies=self._builder_dependencies(),
                    )

                self.assertIsNotNone(legacy_monitor)
                self.assertIsNotNone(internal_monitor)
                assert legacy_monitor is not None
                assert internal_monitor is not None
                self.assertIs(legacy_monitor.coordinator.vision_observer, sentinel_provider)
                self.assertIs(internal_monitor.coordinator.vision_observer, sentinel_provider)
                self.assertIsInstance(legacy_monitor, proactive_service_mod.ProactiveMonitorService)
                self.assertIsInstance(internal_monitor, proactive_service_mod.ProactiveMonitorService)
                self.assertEqual(provider_factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
