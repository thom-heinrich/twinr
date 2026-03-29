from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.audio import AmbientAudioLevelSample, AudioCaptureReadinessProbe
from twinr.hardware.respeaker.models import ReSpeakerSignalSnapshot
from twinr.proactive.runtime.audio_perception import (
    ReSpeakerAudioPerceptionSnapshot,
    derive_respeaker_audio_perception_guard,
    observe_audio_perception_once,
    render_audio_perception_snapshot_lines,
)
from twinr.proactive.runtime.audio_policy import ReSpeakerAudioPolicySnapshot
from twinr.proactive.social import ProactiveAudioSnapshot, SocialAudioObservation


class ReSpeakerAudioPerceptionTests(unittest.TestCase):
    def test_guard_marks_background_media_as_not_device_directed(self) -> None:
        guard = derive_respeaker_audio_perception_guard(
            audio=SocialAudioObservation(
                speech_detected=False,
                background_media_likely=True,
                room_quiet=False,
            ),
            policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=10.0,
                background_media_likely=True,
                speech_delivery_defer_reason="background_media_active",
            ),
        )

        self.assertEqual(guard.room_context, "background_media")
        self.assertFalse(guard.device_directed_speech_candidate)
        self.assertEqual(guard.guard_reason, "background_media_active")

    def test_guard_prefers_non_speech_context_over_overlap_when_classified(self) -> None:
        guard = derive_respeaker_audio_perception_guard(
            audio=SocialAudioObservation(
                speech_detected=True,
                speech_overlap_likely=True,
                non_speech_audio_likely=True,
                room_quiet=False,
            ),
            policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=10.0,
                presence_audio_active=False,
                room_busy_or_overlapping=True,
                non_speech_audio_likely=True,
                speech_delivery_defer_reason="non_speech_audio_active",
                runtime_alert_code="ready",
            ),
        )

        self.assertEqual(guard.room_context, "non_speech_activity")
        self.assertFalse(guard.device_directed_speech_candidate)
        self.assertEqual(guard.guard_reason, "non_speech_audio_active")

    def test_guard_marks_stable_speech_as_device_directed_candidate(self) -> None:
        guard = derive_respeaker_audio_perception_guard(
            audio=SocialAudioObservation(
                speech_detected=True,
                room_quiet=False,
                direction_confidence=0.91,
            ),
            policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=20.0,
                presence_audio_active=True,
                speaker_direction_stable=True,
                runtime_alert_code="ready",
            ),
        )

        self.assertEqual(guard.room_context, "speech")
        self.assertTrue(guard.device_directed_speech_candidate)
        self.assertIsNone(guard.guard_reason)

    def test_render_lines_include_guard_and_runtime_facts(self) -> None:
        snapshot = ReSpeakerAudioPerceptionSnapshot(
            audio_snapshot=ProactiveAudioSnapshot(
                observation=SocialAudioObservation(
                    speech_detected=True,
                    room_quiet=False,
                    non_speech_audio_likely=False,
                    background_media_likely=False,
                    device_runtime_mode="audio_ready",
                    host_control_ready=True,
                ),
                sample=AmbientAudioLevelSample(
                    duration_ms=1000,
                    chunk_count=10,
                    active_chunk_count=5,
                    average_rms=123,
                    peak_rms=456,
                    active_ratio=0.5,
                ),
                signal_snapshot=ReSpeakerSignalSnapshot(
                    captured_at=1.0,
                    source="respeaker",
                    source_type="hardware",
                    sensor_window_ms=1000,
                    device_runtime_mode="audio_ready",
                    host_control_ready=True,
                ),
            ),
            audio_policy_snapshot=ReSpeakerAudioPolicySnapshot(
                observed_at=1.0,
                presence_audio_active=True,
                speaker_direction_stable=True,
                runtime_alert_code="ready",
                runtime_alert_message="ReSpeaker capture and host-control are ready.",
            ),
            perception_guard=derive_respeaker_audio_perception_guard(
                audio=SocialAudioObservation(
                    speech_detected=True,
                    room_quiet=False,
                ),
                policy_snapshot=ReSpeakerAudioPolicySnapshot(
                    observed_at=1.0,
                    presence_audio_active=True,
                    speaker_direction_stable=True,
                    runtime_alert_code="ready",
                ),
            ),
            respeaker_targeted=True,
            capture_probe=AudioCaptureReadinessProbe(
                device="plughw:CARD=Array,DEV=0",
                sample_rate=16000,
                channels=1,
                chunk_ms=100,
                duration_ms=250,
                target_chunk_count=1,
                captured_chunk_count=1,
                captured_bytes=3200,
            ),
        )

        lines = set(render_audio_perception_snapshot_lines(snapshot))

        self.assertIn("proactive_audio_target=respeaker", lines)
        self.assertIn("proactive_audio_capture_probe_ready=true", lines)
        self.assertIn("proactive_audio_policy_runtime_alert_code=ready", lines)
        self.assertIn("proactive_audio_room_context=speech", lines)
        self.assertIn("proactive_device_directed_speech_candidate=true", lines)

    def test_observe_audio_perception_once_skips_pcm_probe_when_voice_orchestrator_owns_device(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr",
            proactive_audio_enabled=True,
            proactive_enabled=True,
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8765/voice",
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
            voice_orchestrator_audio_device="plughw:CARD=Array,DEV=0",
            proactive_audio_sample_ms=1000,
        )
        audio_snapshot = ProactiveAudioSnapshot(
            observation=SocialAudioObservation(
                speech_detected=True,
                room_quiet=False,
                device_runtime_mode="audio_ready",
                host_control_ready=True,
            )
        )

        with (
            patch("twinr.proactive.runtime.audio_perception.config_targets_respeaker", return_value=True),
            patch(
                "twinr.proactive.runtime.audio_perception._proactive_pcm_capture_conflicts_with_voice_orchestrator",
                return_value=True,
            ),
            patch("twinr.proactive.runtime.audio_perception.AmbientAudioSampler.from_config") as sampler_factory,
            patch("twinr.proactive.runtime.audio_perception.AmbientAudioObservationProvider") as fallback_factory,
            patch("twinr.proactive.runtime.audio_perception.ReSpeakerAudioObservationProvider") as observer_factory,
            patch("twinr.proactive.runtime.audio_perception.ReSpeakerSignalProvider"),
            patch("twinr.proactive.runtime.audio_perception.ReSpeakerAudioPolicyTracker.from_config") as tracker_factory,
        ):
            observer_factory.return_value.observe.return_value = audio_snapshot
            tracker_factory.return_value.observe.return_value = ReSpeakerAudioPolicySnapshot(
                observed_at=1.0,
                presence_audio_active=True,
                speaker_direction_stable=True,
                runtime_alert_code="ready",
            )

            snapshot = observe_audio_perception_once(config)

        sampler_factory.assert_not_called()
        fallback_factory.assert_not_called()
        self.assertTrue(snapshot.respeaker_targeted)
        self.assertIsNone(snapshot.capture_probe)
        self.assertTrue(snapshot.audio_snapshot.observation.speech_detected)

    def test_observe_audio_perception_once_uses_pcm_probe_when_voice_orchestrator_inactive(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr",
            proactive_audio_enabled=True,
            proactive_enabled=True,
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://127.0.0.1:8765/voice",
            audio_input_device="plughw:CARD=Array,DEV=0",
            proactive_audio_input_device="plughw:CARD=Array,DEV=0",
            voice_orchestrator_audio_device="plughw:CARD=Array,DEV=0",
            proactive_audio_sample_ms=1000,
        )
        sampler = unittest.mock.Mock()
        sampler.chunk_ms = 100
        probe = AudioCaptureReadinessProbe(
            device="plughw:CARD=Array,DEV=0",
            sample_rate=16000,
            channels=1,
            chunk_ms=100,
            duration_ms=250,
            target_chunk_count=1,
            captured_chunk_count=1,
            captured_bytes=3200,
        )
        sampler.require_readable_frames.return_value = probe
        audio_snapshot = ProactiveAudioSnapshot(
            observation=SocialAudioObservation(
                speech_detected=False,
                room_quiet=True,
                device_runtime_mode="audio_ready",
                host_control_ready=True,
            )
        )

        with (
            patch("twinr.proactive.runtime.audio_perception.config_targets_respeaker", return_value=True),
            patch(
                "twinr.proactive.runtime.audio_perception._proactive_pcm_capture_conflicts_with_voice_orchestrator",
                return_value=False,
            ),
            patch("twinr.proactive.runtime.audio_perception.AmbientAudioSampler.from_config", return_value=sampler) as sampler_factory,
            patch("twinr.proactive.runtime.audio_perception.AmbientAudioObservationProvider") as fallback_factory,
            patch("twinr.proactive.runtime.audio_perception.ReSpeakerAudioObservationProvider") as observer_factory,
            patch("twinr.proactive.runtime.audio_perception.ReSpeakerSignalProvider"),
            patch("twinr.proactive.runtime.audio_perception.ReSpeakerAudioPolicyTracker.from_config") as tracker_factory,
        ):
            observer_factory.return_value.observe.return_value = audio_snapshot
            tracker_factory.return_value.observe.return_value = ReSpeakerAudioPolicySnapshot(
                observed_at=1.0,
                presence_audio_active=False,
                speaker_direction_stable=False,
                runtime_alert_code="ready",
            )

            snapshot = observe_audio_perception_once(config)

        sampler_factory.assert_called_once_with(config)
        sampler.require_readable_frames.assert_called_once()
        fallback_factory.assert_called_once()
        self.assertIs(snapshot.capture_probe, probe)
        self.assertTrue(snapshot.respeaker_targeted)


if __name__ == "__main__":
    unittest.main()
