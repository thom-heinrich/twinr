from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupLane
from twinr.proactive.social.engine import SocialFineHandGesture, SocialVisionObservation


class GestureWakeupLaneTests(unittest.TestCase):
    def test_lane_triggers_for_configured_peace_sign(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(TwinrConfig(project_root=temp_dir))

            first = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.91,
                ),
            )
            decision = lane.observe(
                observed_at=11.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.91,
                ),
            )

        self.assertFalse(first.active)
        self.assertEqual(first.reason, "awaiting_gesture_wakeup_visibility")
        self.assertTrue(decision.active)
        self.assertEqual(decision.trigger_gesture, SocialFineHandGesture.PEACE_SIGN)
        self.assertEqual(decision.request_source, "gesture")

    def test_lane_ignores_other_supported_gestures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.95,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_gesture_wakeup_candidate")

    def test_lane_applies_same_gesture_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(
                TwinrConfig(project_root=temp_dir, gesture_wakeup_cooldown_s=2.5)
            )

            first = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.91,
                ),
            )
            second = lane.observe(
                observed_at=11.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.91,
                ),
            )
            third = lane.observe(
                observed_at=11.4,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.91,
                ),
            )

        self.assertFalse(first.active)
        self.assertEqual(first.reason, "awaiting_gesture_wakeup_visibility")
        self.assertTrue(second.active)
        self.assertFalse(third.active)
        self.assertEqual(third.reason, "gesture_wakeup_cooldown")

    def test_lane_can_be_disabled_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(
                TwinrConfig(project_root=temp_dir, gesture_wakeup_enabled=False)
            )

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.95,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "gesture_wakeup_disabled")


if __name__ == "__main__":
    unittest.main()
