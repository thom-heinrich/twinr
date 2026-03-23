from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive.runtime.safety_trigger_fusion import SafetyTriggerFusionBridge
from twinr.proactive.social.engine import (
    SocialAudioObservation,
    SocialBodyPose,
    SocialMotionState,
    SocialObservation,
    SocialTriggerEngine,
    SocialVisionObservation,
)


class SafetyTriggerFusionBridgeTests(unittest.TestCase):
    def test_prefers_fused_possible_fall_for_short_window_floor_sequence(self) -> None:
        config = TwinrConfig(proactive_enabled=True)
        engine = SocialTriggerEngine.from_config(config)
        bridge = SafetyTriggerFusionBridge.from_config(config, engine=engine)

        bridge.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=True,
                low_motion=False,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        bridge.observe(
            SocialObservation(
                observed_at=2.0,
                inspected=True,
                low_motion=False,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.FLOOR,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        decision = bridge.observe(
            SocialObservation(
                observed_at=5.5,
                inspected=True,
                low_motion=False,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.FLOOR,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "floor_stillness")
        self.assertEqual(bridge.last_selected_source, "event_fusion")
        self.assertIsNotNone(bridge.last_selected_claim)
        self.assertEqual(bridge.last_selected_claim.state, "floor_stillness_after_drop")

    def test_uses_fused_distress_possible_from_recent_slumped_sequence_and_audio(self) -> None:
        config = TwinrConfig(proactive_enabled=True)
        engine = SocialTriggerEngine.from_config(config)
        bridge = SafetyTriggerFusionBridge.from_config(config, engine=engine)

        bridge.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=True,
                low_motion=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        bridge.observe(
            SocialObservation(
                observed_at=4.5,
                inspected=True,
                low_motion=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        decision = bridge.observe(
            SocialObservation(
                observed_at=4.6,
                inspected=False,
                low_motion=True,
                audio=SocialAudioObservation(
                    speech_detected=True,
                    distress_detected=True,
                ),
            )
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "distress_possible")
        self.assertEqual(bridge.last_selected_source, "event_fusion")
        self.assertIsNotNone(bridge.last_selected_claim)
        self.assertEqual(bridge.last_selected_claim.state, "distress_possible")

    def test_fused_distress_possible_respects_engine_cooldown_after_dispatch(self) -> None:
        config = TwinrConfig(proactive_enabled=True)
        engine = SocialTriggerEngine.from_config(config)
        bridge = SafetyTriggerFusionBridge.from_config(config, engine=engine)

        bridge.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=True,
                low_motion=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        bridge.observe(
            SocialObservation(
                observed_at=4.5,
                inspected=True,
                low_motion=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        first = bridge.observe(
            SocialObservation(
                observed_at=4.6,
                inspected=False,
                low_motion=True,
                audio=SocialAudioObservation(
                    speech_detected=True,
                    distress_detected=True,
                ),
            )
        )
        second = bridge.observe(
            SocialObservation(
                observed_at=5.0,
                inspected=False,
                low_motion=True,
                audio=SocialAudioObservation(
                    speech_detected=True,
                    distress_detected=True,
                ),
            )
        )

        self.assertIsNotNone(first)
        self.assertEqual(first.trigger_id, "distress_possible")
        self.assertIsNone(second)
        self.assertEqual(bridge.best_fused_evaluation.blocked_reason, "cooldown_active")

    def test_falls_back_to_engine_for_visibility_loss_possible_fall_case(self) -> None:
        config = TwinrConfig(
            proactive_enabled=True,
            proactive_possible_fall_stillness_s=4.0,
            proactive_possible_fall_visibility_loss_hold_s=4.0,
            proactive_possible_fall_visibility_loss_arming_s=2.0,
            proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
            proactive_possible_fall_score_threshold=0.65,
        )
        engine = SocialTriggerEngine.from_config(config)
        bridge = SafetyTriggerFusionBridge.from_config(config, engine=engine)

        bridge.observe(
            SocialObservation(
                observed_at=0.0,
                inspected=True,
                pir_motion_detected=True,
                low_motion=False,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        bridge.observe(
            SocialObservation(
                observed_at=2.5,
                inspected=True,
                pir_motion_detected=True,
                low_motion=False,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        bridge.observe(
            SocialObservation(
                observed_at=2.55,
                inspected=False,
                pir_motion_detected=False,
                low_motion=True,
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        decision = bridge.observe(
            SocialObservation(
                observed_at=7.6,
                inspected=False,
                pir_motion_detected=False,
                low_motion=True,
                audio=SocialAudioObservation(speech_detected=False),
            )
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "possible_fall")
        self.assertEqual(bridge.last_selected_source, "social_engine")


if __name__ == "__main__":
    unittest.main()
