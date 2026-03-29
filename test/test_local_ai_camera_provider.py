from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.hardware.ai_camera import (
    AICameraBodyPose,
    AICameraBox,
    AICameraFineHandGesture,
    AICameraGestureEvent,
    AICameraMotionState,
    AICameraObjectDetection,
    AICameraObservation,
    AICameraZone,
)
from twinr.proactive.social.engine import (
    SocialBodyPose,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
)
from twinr.proactive.social.local_camera_provider import LocalAICameraObservationProvider
from twinr.proactive.social.perception_stream import (
    attention_stream_authoritative,
    gesture_stream_authoritative,
)


class _FakeAdapter:
    def __init__(self, observation):
        self.observation = observation
        self.last_gesture_kwargs = None
        self.attention_stream_calls = 0
        self.gesture_stream_calls = 0
        self.perception_stream_calls = 0
        self._last_attention_debug_details = None
        self._last_gesture_debug_details = None

    def observe(self):
        if isinstance(self.observation, Exception):
            raise self.observation
        return self.observation

    def observe_attention(self):
        return self.observe()

    def observe_attention_stream(self):
        self.attention_stream_calls += 1
        return self.observe()

    def observe_gesture(self, **kwargs):
        self.last_gesture_kwargs = dict(kwargs)
        return self.observe()

    def observe_gesture_stream(self):
        self.gesture_stream_calls += 1
        return self.observe()

    def observe_perception_stream(self):
        self.perception_stream_calls += 1
        return self.observe()

    def get_last_attention_debug_details(self):
        return self._last_attention_debug_details

    def get_last_gesture_debug_details(self):
        return self._last_gesture_debug_details


class LocalAICameraProviderTests(unittest.TestCase):
    def test_provider_maps_local_camera_observation_to_social_contract(self) -> None:
        provider = LocalAICameraObservationProvider(
            adapter=_FakeAdapter(
                AICameraObservation(
                    observed_at=100.0,
                    camera_online=True,
                    camera_ready=True,
                    camera_ai_ready=True,
                    camera_error=None,
                    last_camera_frame_at=99.5,
                    last_camera_health_change_at=98.0,
                    person_count=2,
                    primary_person_box=AICameraBox(top=0.1, left=0.2, bottom=0.8, right=0.7),
                    primary_person_zone=AICameraZone.CENTER,
                    looking_toward_device=True,
                    person_near_device=True,
                    engaged_with_device=True,
                    visual_attention_score=0.81,
                    body_pose=AICameraBodyPose.SEATED,
                    pose_confidence=0.67,
                    motion_state=AICameraMotionState.APPROACHING,
                    motion_confidence=0.63,
                    hand_or_object_near_camera=True,
                    showing_intent_likely=True,
                    gesture_event=AICameraGestureEvent.TWO_HAND_DISMISS,
                    gesture_confidence=0.76,
                    fine_hand_gesture=AICameraFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.83,
                    objects=(
                        AICameraObjectDetection(
                            label="cup",
                            confidence=0.92,
                            zone=AICameraZone.RIGHT,
                            box=AICameraBox(top=0.3, left=0.6, bottom=0.7, right=0.9),
                        ),
                    ),
                )
            )
        )

        snapshot = provider.observe()

        self.assertEqual(snapshot.model, "local-imx500")
        self.assertIn("provider=local_ai_camera", snapshot.response_text)
        self.assertTrue(snapshot.observation.person_visible)
        self.assertEqual(snapshot.observation.person_count, 2)
        self.assertEqual(snapshot.observation.primary_person_zone, SocialPersonZone.CENTER)
        self.assertAlmostEqual(snapshot.observation.primary_person_center_x or 0.0, 0.45, places=3)
        self.assertTrue(snapshot.observation.looking_toward_device)
        self.assertTrue(snapshot.observation.person_near_device)
        self.assertTrue(snapshot.observation.engaged_with_device)
        self.assertAlmostEqual(snapshot.observation.visual_attention_score or 0.0, 0.81, places=3)
        self.assertEqual(snapshot.observation.body_pose, SocialBodyPose.SEATED)
        self.assertEqual(snapshot.observation.motion_state, SocialMotionState.APPROACHING)
        self.assertAlmostEqual(snapshot.observation.motion_confidence or 0.0, 0.63, places=3)
        self.assertTrue(snapshot.observation.hand_or_object_near_camera)
        self.assertTrue(snapshot.observation.showing_intent_likely)
        self.assertEqual(snapshot.observation.coarse_arm_gesture, SocialGestureEvent.TWO_HAND_DISMISS)
        self.assertEqual(snapshot.observation.gesture_event, SocialGestureEvent.TWO_HAND_DISMISS)
        self.assertEqual(snapshot.observation.fine_hand_gesture, SocialFineHandGesture.THUMBS_UP)
        self.assertAlmostEqual(snapshot.observation.fine_hand_gesture_confidence or 0.0, 0.83, places=3)
        self.assertEqual(len(snapshot.observation.objects), 1)
        self.assertEqual(snapshot.observation.objects[0].label, "cup")
        self.assertEqual(snapshot.observation.objects[0].zone, SocialPersonZone.RIGHT)
        self.assertTrue(snapshot.observation.camera_ready)

    def test_provider_degrades_to_health_snapshot_when_adapter_raises(self) -> None:
        provider = LocalAICameraObservationProvider(adapter=_FakeAdapter(RuntimeError("boom")))

        snapshot = provider.observe()

        self.assertFalse(snapshot.observation.person_visible)
        self.assertFalse(snapshot.observation.camera_online)
        self.assertFalse(snapshot.observation.camera_ready)
        self.assertFalse(snapshot.observation.camera_ai_ready)
        self.assertEqual(snapshot.observation.camera_error, "local_ai_camera_provider_failed")

    def test_provider_uses_current_frame_fast_path_for_gesture_snapshots(self) -> None:
        adapter = _FakeAdapter(
            AICameraObservation(
                observed_at=10.0,
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                fine_hand_gesture=AICameraFineHandGesture.PEACE_SIGN,
                fine_hand_gesture_confidence=0.84,
                hand_or_object_near_camera=True,
                gesture_temporal_authoritative=True,
                gesture_activation_key="fine:peace_sign",
                gesture_activation_token=4,
                gesture_activation_started_at=9.9,
                gesture_activation_changed_at=10.0,
                gesture_activation_source="live_stream",
                gesture_activation_rising=True,
            )
        )
        provider = LocalAICameraObservationProvider(adapter=adapter)

        snapshot = provider.observe_gesture()

        self.assertEqual(snapshot.observation.fine_hand_gesture, SocialFineHandGesture.PEACE_SIGN)
        self.assertTrue(gesture_stream_authoritative(snapshot.observation))
        self.assertEqual(adapter.gesture_stream_calls, 1)
        self.assertIsNone(adapter.last_gesture_kwargs)

    def test_provider_prefers_explicit_stream_attention_path_when_available(self) -> None:
        adapter = _FakeAdapter(
            AICameraObservation(
                observed_at=12.0,
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                person_count=1,
            )
        )
        provider = LocalAICameraObservationProvider(adapter=adapter)

        snapshot = provider.observe_attention()

        self.assertTrue(snapshot.observation.camera_ready)
        self.assertTrue(attention_stream_authoritative(snapshot.observation))
        self.assertEqual(adapter.attention_stream_calls, 1)

    def test_provider_exposes_combined_perception_stream_from_one_adapter_call(self) -> None:
        adapter = _FakeAdapter(
            AICameraObservation(
                observed_at=12.0,
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                person_count=1,
                looking_toward_device=True,
                looking_signal_state="confirmed",
                looking_signal_source="face_anchor_matched",
                visual_attention_score=0.81,
                hand_or_object_near_camera=True,
                fine_hand_gesture=AICameraFineHandGesture.THUMBS_UP,
                fine_hand_gesture_confidence=0.92,
                gesture_temporal_authoritative=True,
                gesture_activation_key="fine:thumbs_up",
                gesture_activation_token=7,
                gesture_activation_started_at=11.8,
                gesture_activation_changed_at=12.0,
                gesture_activation_source="live_stream",
                gesture_activation_rising=True,
            )
        )
        adapter._last_attention_debug_details = {
            "stream_mode": "attention_stream",
            "attention_instant_looking_toward_device": True,
            "attention_instant_visual_attention_score": 0.81,
            "attention_instant_looking_signal_state": "confirmed",
            "attention_instant_looking_signal_source": "face_anchor_matched",
            "attention_stream_changed": True,
        }
        adapter._last_gesture_debug_details = {
            "stream_mode": "gesture_stream",
            "live_fine_hand_gesture": "thumbs_up",
            "live_fine_hand_gesture_confidence": 0.92,
            "live_gesture_event": "none",
            "live_gesture_confidence": None,
            "gesture_stream_temporal_reason": "live_passthrough",
            "gesture_stream_resolved_source": "live_stream",
            "authoritative_gesture_rising": True,
        }
        provider = LocalAICameraObservationProvider(adapter=adapter)

        snapshot = provider.observe_perception_stream()

        self.assertEqual(adapter.perception_stream_calls, 1)
        self.assertTrue(attention_stream_authoritative(snapshot.observation))
        self.assertTrue(gesture_stream_authoritative(snapshot.observation))

    def test_provider_reads_adapter_debug_snapshots_for_combined_perception_stream(self) -> None:
        adapter = _FakeAdapter(
            AICameraObservation(
                observed_at=14.0,
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                person_count=1,
                looking_toward_device=True,
                looking_signal_state="confirmed",
                looking_signal_source="face_anchor_matched",
                visual_attention_score=0.77,
                hand_or_object_near_camera=True,
                fine_hand_gesture=AICameraFineHandGesture.PEACE_SIGN,
                fine_hand_gesture_confidence=0.88,
                gesture_temporal_authoritative=True,
                gesture_activation_key="fine:peace_sign",
                gesture_activation_token=9,
                gesture_activation_started_at=13.7,
                gesture_activation_changed_at=14.0,
                gesture_activation_source="temporal_consensus",
                gesture_activation_rising=False,
            )
        )
        provider = LocalAICameraObservationProvider(adapter=adapter)
        adapter._last_attention_debug_details = {
            "stream_mode": "attention_stream",
            "attention_instant_looking_toward_device": True,
            "attention_instant_visual_attention_score": 0.77,
            "attention_instant_looking_signal_state": "confirmed",
            "attention_instant_looking_signal_source": "face_anchor_matched",
            "attention_stream_candidate_state": "confirmed",
            "attention_stream_candidate_source": "face_anchor_matched",
            "attention_stream_changed": False,
        }
        adapter._last_gesture_debug_details = {
            "stream_mode": "gesture_stream",
            "live_fine_hand_gesture": "peace_sign",
            "live_fine_hand_gesture_confidence": 0.88,
            "live_gesture_event": "none",
            "live_gesture_confidence": None,
            "gesture_stream_temporal_reason": "consensus",
            "gesture_stream_resolved_source": "temporal_consensus",
        }

        snapshot = provider.observe_perception_stream()
        stream = snapshot.observation.perception_stream

        self.assertIsNotNone(stream)
        assert stream is not None
        self.assertEqual(stream.attention.instant_signal_source, "face_anchor_matched")
        self.assertEqual(stream.gesture.activation_token, 9)
        self.assertEqual(stream.gesture.temporal_reason, "consensus")


if __name__ == "__main__":
    unittest.main()
