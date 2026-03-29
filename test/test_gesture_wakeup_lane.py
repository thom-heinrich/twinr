from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.gesture_wakeup_lane import GestureWakeupLane
from twinr.proactive.social.engine import SocialFineHandGesture, SocialVisionObservation
from twinr.proactive.social.perception_stream import (
    PerceptionGestureStreamObservation,
    PerceptionStreamObservation,
)


def _authoritative_observation(
    *,
    token: int | None,
    gesture: SocialFineHandGesture,
    confidence: float,
) -> SocialVisionObservation:
    return SocialVisionObservation(
        hand_or_object_near_camera=True,
        fine_hand_gesture=gesture,
        fine_hand_gesture_confidence=confidence,
        perception_stream=PerceptionStreamObservation(
            gesture=PerceptionGestureStreamObservation(
                authoritative=True,
                activation_key=None if token is None else f"fine:{gesture.value}",
                activation_token=token,
                activation_rising=token is not None,
            )
        ),
    )


class GestureWakeupLaneTests(unittest.TestCase):
    def test_lane_requires_authoritative_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                    fine_hand_gesture_confidence=0.91,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "gesture_wakeup_not_authoritative")

    def test_lane_triggers_for_configured_authoritative_peace_sign(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=1,
                    gesture=SocialFineHandGesture.PEACE_SIGN,
                    confidence=0.55,
                ),
            )

        self.assertTrue(decision.active)
        self.assertEqual(decision.reason, "gesture_wakeup:peace_sign")
        self.assertEqual(decision.request_source, "gesture")

    def test_lane_ignores_other_authoritative_gestures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=2,
                    gesture=SocialFineHandGesture.THUMBS_UP,
                    confidence=0.95,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "no_gesture_wakeup_candidate")

    def test_lane_applies_same_gesture_cooldown_across_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(
                TwinrConfig(project_root=temp_dir, gesture_wakeup_cooldown_s=2.5)
            )

            first = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=1,
                    gesture=SocialFineHandGesture.PEACE_SIGN,
                    confidence=0.91,
                ),
            )
            held = lane.observe(
                observed_at=10.2,
                observation=_authoritative_observation(
                    token=1,
                    gesture=SocialFineHandGesture.PEACE_SIGN,
                    confidence=0.91,
                ),
            )
            blocked_repeat = lane.observe(
                observed_at=10.5,
                observation=_authoritative_observation(
                    token=2,
                    gesture=SocialFineHandGesture.PEACE_SIGN,
                    confidence=0.91,
                ),
            )

        self.assertTrue(first.active)
        self.assertFalse(held.active)
        self.assertEqual(held.reason, "gesture_wakeup_already_active")
        self.assertFalse(blocked_repeat.active)
        self.assertEqual(blocked_repeat.reason, "gesture_wakeup_cooldown")

    def test_lane_can_be_disabled_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureWakeupLane.from_config(
                TwinrConfig(project_root=temp_dir, gesture_wakeup_enabled=False)
            )

            decision = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=3,
                    gesture=SocialFineHandGesture.PEACE_SIGN,
                    confidence=0.95,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "gesture_wakeup_disabled")


if __name__ == "__main__":
    unittest.main()
