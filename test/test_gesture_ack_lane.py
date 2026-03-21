from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive.runtime.gesture_ack_lane import GestureAckLane
from twinr.proactive.social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation


class GestureAckLaneTests(unittest.TestCase):
    def test_lane_acknowledges_supported_fine_gesture_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.94,
                ),
            )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "thumbs_up")

    def test_lane_ignores_open_palm_in_dedicated_user_gesture_hot_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.OPEN_PALM,
                    fine_hand_gesture_confidence=0.98,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_supported_live_gesture")

    def test_lane_applies_same_symbol_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            first = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.94,
                ),
            )
            second = lane.observe(
                observed_at=10.1,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.94,
                ),
            )

        self.assertTrue(first.active)
        self.assertFalse(second.active)
        self.assertEqual(second.reason, "live_gesture_cooldown")

    def test_lane_acknowledges_wave_from_supported_coarse_gesture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    gesture_event=SocialGestureEvent.WAVE,
                    gesture_confidence=0.86,
                ),
            )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "waving_hand")


if __name__ == "__main__":
    unittest.main()
