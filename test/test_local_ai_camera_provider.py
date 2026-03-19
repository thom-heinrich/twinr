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


class _FakeAdapter:
    def __init__(self, observation):
        self.observation = observation

    def observe(self):
        if isinstance(self.observation, Exception):
            raise self.observation
        return self.observation


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


if __name__ == "__main__":
    unittest.main()
