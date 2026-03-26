from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import sys
import unittest
from unittest.mock import patch

# pylint: disable=no-member

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupDecision
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.runtime.service import (
    ProactiveCoordinator,
    build_default_proactive_monitor,
    _proactive_pcm_capture_conflicts_with_voice_orchestrator,
    _voice_orchestrator_capture_device,
)
from twinr.proactive.social.engine import (
    SocialAudioObservation,
    SocialFineHandGesture,
    SocialTriggerDecision,
    SocialTriggerPriority,
    SocialVisionObservation,
)
from twinr.proactive.social.observers import ProactiveAudioSnapshot


class ProactiveCoordinatorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
