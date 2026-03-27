from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import sys
import unittest
from unittest.mock import patch

# mypy: ignore-errors

# pylint: disable=no-member

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupDecision
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime import service as proactive_service_mod
from twinr.proactive.runtime.service import (
    ProactiveCoordinator,
    build_default_proactive_monitor,
    _proactive_pcm_capture_conflicts_with_voice_orchestrator,
    _voice_orchestrator_capture_device,
)
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
            remote_proxy_vision_provider_cls=proactive_service_mod.RemoteAICameraObservationProvider,
            remote_frame_vision_provider_cls=proactive_service_mod.RemoteFrameAICameraObservationProvider,
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

    def test_build_default_monitor_uses_remote_camera_provider_when_configured(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="remote_proxy",
            proactive_remote_camera_base_url="http://10.42.0.2:8767",
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )
        sentinel_provider = object()

        with (
            patch(
                "twinr.proactive.runtime.service.RemoteAICameraObservationProvider.from_config",
                return_value=sentinel_provider,
            ) as remote_factory,
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
        remote_factory.assert_called_once_with(config)
        local_factory.assert_not_called()
        openai_provider.assert_not_called()

    def test_build_default_monitor_uses_remote_frame_provider_when_configured(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            proactive_vision_provider="remote_frame",
            proactive_remote_camera_base_url="http://10.42.0.2:8767",
        )
        runtime = SimpleNamespace(
            ops_events=SimpleNamespace(append=lambda **_kwargs: None),
            fail=lambda _detail: None,
            status=SimpleNamespace(value="waiting"),
        )
        sentinel_provider = object()

        with (
            patch(
                "twinr.proactive.runtime.service.RemoteFrameAICameraObservationProvider.from_config",
                return_value=sentinel_provider,
            ) as remote_frame_factory,
            patch(
                "twinr.proactive.runtime.service.RemoteAICameraObservationProvider.from_config",
            ) as remote_proxy_factory,
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
        remote_frame_factory.assert_called_once_with(config)
        remote_proxy_factory.assert_not_called()
        local_factory.assert_not_called()
        openai_provider.assert_not_called()

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

    def test_wrapper_builder_matches_internal_builder_for_provider_variants(self) -> None:
        cases = [
            (
                "remote_proxy",
                {
                    "display_driver": "hdmi_wayland",
                    "proactive_vision_provider": "remote_proxy",
                    "proactive_remote_camera_base_url": "http://10.42.0.2:8767",
                },
                "twinr.proactive.runtime.service.RemoteAICameraObservationProvider.from_config",
            ),
            (
                "remote_frame",
                {
                    "display_driver": "hdmi_wayland",
                    "proactive_vision_provider": "remote_frame",
                    "proactive_remote_camera_base_url": "http://10.42.0.2:8767",
                },
                "twinr.proactive.runtime.service.RemoteFrameAICameraObservationProvider.from_config",
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
