from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.proactive.runtime.display_attention_camera_fusion import DisplayAttentionCameraFusion
from twinr.proactive.social.engine import (
    SocialBodyPose,
    SocialFineHandGesture,
    SocialMotionState,
    SocialPersonZone,
    SocialVisionObservation,
)


class DisplayAttentionCameraFusionTests(unittest.TestCase):
    def test_fuse_attention_keeps_current_attention_truth_while_overlaying_pose_and_gesture_semantics(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            display_attention_refresh_interval_s=0.5,
            proactive_capture_interval_s=6.0,
        )
        fusion = DisplayAttentionCameraFusion.from_config(config)
        fusion.remember_full(
            observed_at=10.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_center_x=0.51,
                primary_person_center_y=0.49,
                looking_toward_device=True,
                person_near_device=True,
                engaged_with_device=True,
                visual_attention_score=0.82,
                body_pose=SocialBodyPose.UPRIGHT,
                motion_state=SocialMotionState.STILL,
            ),
        )
        fusion.remember_gesture(
            observed_at=10.2,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_center_x=0.53,
                primary_person_center_y=0.5,
                hand_or_object_near_camera=True,
                showing_intent_likely=True,
                fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
            ),
        )

        result = fusion.fuse_attention(
            observed_at=10.4,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_center_x=0.52,
                primary_person_center_y=0.5,
                looking_toward_device=False,
                hand_or_object_near_camera=False,
                showing_intent_likely=False,
            ),
        )

        self.assertFalse(result.observation.looking_toward_device)
        self.assertIsNone(result.observation.person_near_device)
        self.assertIsNone(result.observation.engaged_with_device)
        self.assertIsNone(result.observation.visual_attention_score)
        self.assertEqual(result.observation.body_pose, SocialBodyPose.UPRIGHT)
        self.assertEqual(result.observation.motion_state, SocialMotionState.STILL)
        self.assertTrue(result.observation.hand_or_object_near_camera)
        self.assertTrue(result.observation.showing_intent_likely)
        self.assertEqual(result.observation.fine_hand_gesture, SocialFineHandGesture.THUMBS_UP)
        self.assertEqual(result.debug_details["used_pose_source"], "full_observe")
        self.assertEqual(result.debug_details["used_hand_source"], "gesture_refresh")

    def test_fuse_attention_holds_recent_visible_person_across_short_dropout(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            display_attention_refresh_interval_s=0.5,
            proactive_capture_interval_s=6.0,
        )
        fusion = DisplayAttentionCameraFusion.from_config(config)
        first = fusion.fuse_attention(
            observed_at=20.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.RIGHT,
                primary_person_center_x=0.76,
                primary_person_center_y=0.44,
                looking_toward_device=True,
            ),
        )
        self.assertTrue(first.observation.person_visible)

        dropout = fusion.fuse_attention(
            observed_at=20.6,
            observation=SocialVisionObservation(
                person_visible=False,
                person_count=0,
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=20.6,
            ),
        )

        self.assertTrue(dropout.observation.person_visible)
        self.assertEqual(dropout.observation.primary_person_zone, SocialPersonZone.RIGHT)
        self.assertTrue(dropout.observation.looking_toward_device)
        self.assertEqual(dropout.debug_details["dropout_hold_source"], "attention_fused")

    def test_recent_hand_semantics_tracks_fresh_attention_or_gesture_context(self) -> None:
        config = TwinrConfig(
            display_driver="hdmi_wayland",
            display_attention_refresh_interval_s=0.5,
            proactive_capture_interval_s=6.0,
        )
        fusion = DisplayAttentionCameraFusion.from_config(config)

        fusion.fuse_attention(
            observed_at=30.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_center_x=0.5,
                primary_person_center_y=0.5,
                hand_or_object_near_camera=True,
                showing_intent_likely=True,
            ),
        )

        self.assertTrue(fusion.has_recent_hand_semantics(observed_at=30.2))
        self.assertFalse(fusion.has_recent_hand_semantics(observed_at=33.0))


if __name__ == "__main__":
    unittest.main()
