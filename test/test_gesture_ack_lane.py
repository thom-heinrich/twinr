from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive.runtime.gesture_ack_lane import GestureAckLane
from twinr.proactive.social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation


class GestureAckLaneTests(unittest.TestCase):
    def test_lane_honors_current_pi_frozen_ack_floors_for_supported_fine_gestures(self) -> None:
        cases = (
            (SocialFineHandGesture.THUMBS_UP, 0.56, "thumbs_up"),
            (SocialFineHandGesture.THUMBS_DOWN, 0.67, "thumbs_down"),
            (SocialFineHandGesture.POINTING, 0.66, "pointing_hand"),
            (SocialFineHandGesture.PEACE_SIGN, 0.60, "victory_hand"),
            (SocialFineHandGesture.OK_SIGN, 0.86, "ok_hand"),
            (SocialFineHandGesture.MIDDLE_FINGER, 0.90, "warning"),
        )

        for gesture, confidence, expected_symbol in cases:
            with self.subTest(gesture=gesture.value, confidence=confidence):
                with tempfile.TemporaryDirectory() as temp_dir:
                    lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

                    decision = lane.observe(
                        observed_at=10.0,
                        observation=SocialVisionObservation(
                            fine_hand_gesture=gesture,
                            fine_hand_gesture_confidence=confidence,
                        ),
                    )

                self.assertTrue(decision.active)
                self.assertEqual(decision.symbol.value, expected_symbol)

    def test_lane_blocks_values_just_below_current_pi_frozen_ack_floors(self) -> None:
        cases = (
            (SocialFineHandGesture.THUMBS_UP, 0.559),
            (SocialFineHandGesture.THUMBS_DOWN, 0.669),
            (SocialFineHandGesture.POINTING, 0.659),
            (SocialFineHandGesture.PEACE_SIGN, 0.599),
            (SocialFineHandGesture.OK_SIGN, 0.859),
            (SocialFineHandGesture.MIDDLE_FINGER, 0.899),
        )

        for gesture, confidence in cases:
            with self.subTest(gesture=gesture.value, confidence=confidence):
                with tempfile.TemporaryDirectory() as temp_dir:
                    lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

                    decision = lane.observe(
                        observed_at=10.0,
                        observation=SocialVisionObservation(
                            fine_hand_gesture=gesture,
                            fine_hand_gesture_confidence=confidence,
                        ),
                    )

                self.assertFalse(decision.active)
                self.assertEqual(decision.reason, "no_supported_live_gesture")

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

    def test_lane_honors_current_pi_frozen_wave_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    gesture_event=SocialGestureEvent.WAVE,
                    gesture_confidence=0.68,
                ),
            )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "waving_hand")

    def test_lane_blocks_wave_just_below_current_pi_frozen_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    gesture_event=SocialGestureEvent.WAVE,
                    gesture_confidence=0.679,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_supported_live_gesture")

    def test_lane_acknowledges_pi_range_peace_sign(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.661,
                ),
            )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "victory_hand")

    def test_lane_acknowledges_pi_range_thumbs_up_above_ack_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.585,
                ),
            )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "thumbs_up")

    def test_lane_keeps_weak_thumbs_up_below_ack_floor_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.534,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_supported_live_gesture")

    def test_lane_keeps_weak_pose_fallback_pointing_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.POINTING,
                    fine_hand_gesture_confidence=0.575,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_supported_live_gesture")


if __name__ == "__main__":
    unittest.main()
