from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.gesture_ack_lane import GestureAckLane
from twinr.proactive.social.engine import SocialFineHandGesture, SocialGestureEvent, SocialVisionObservation
from twinr.proactive.social.perception_stream import (
    PerceptionGestureStreamObservation,
    PerceptionStreamObservation,
)


def _authoritative_observation(
    *,
    token: int | None,
    fine: SocialFineHandGesture = SocialFineHandGesture.NONE,
    fine_confidence: float | None = None,
    coarse: SocialGestureEvent = SocialGestureEvent.NONE,
    coarse_confidence: float | None = None,
) -> SocialVisionObservation:
    return SocialVisionObservation(
        hand_or_object_near_camera=True,
        fine_hand_gesture=fine,
        fine_hand_gesture_confidence=fine_confidence,
        gesture_event=coarse,
        gesture_confidence=coarse_confidence,
        perception_stream=PerceptionStreamObservation(
            gesture=PerceptionGestureStreamObservation(
                authoritative=True,
                activation_key=(
                    None
                    if token is None
                    else f"fine:{fine.value}"
                    if fine != SocialFineHandGesture.NONE
                    else f"coarse:{coarse.value}"
                ),
                activation_token=token,
                activation_rising=token is not None,
            )
        ),
    )


class GestureAckLaneTests(unittest.TestCase):
    def test_lane_requires_authoritative_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=SocialVisionObservation(
                    fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                    fine_hand_gesture_confidence=0.94,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "gesture_stream_not_authoritative")

    def test_lane_emits_once_per_authoritative_activation_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            first = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=1,
                    fine=SocialFineHandGesture.THUMBS_UP,
                    fine_confidence=0.52,
                ),
            )
            held = lane.observe(
                observed_at=10.2,
                observation=_authoritative_observation(
                    token=1,
                    fine=SocialFineHandGesture.THUMBS_UP,
                    fine_confidence=0.48,
                ),
            )

        self.assertTrue(first.active)
        self.assertEqual(first.symbol.value, "thumbs_up")
        self.assertFalse(held.active)
        self.assertEqual(held.reason, "live_gesture_already_active")

    def test_lane_applies_repeat_hold_across_new_tokens_for_same_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            emitted = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=1,
                    fine=SocialFineHandGesture.PEACE_SIGN,
                    fine_confidence=0.60,
                ),
            )
            cooldown = lane.observe(
                observed_at=10.2,
                observation=_authoritative_observation(
                    token=2,
                    fine=SocialFineHandGesture.PEACE_SIGN,
                    fine_confidence=0.61,
                ),
            )
            still_same = lane.observe(
                observed_at=10.3,
                observation=_authoritative_observation(
                    token=2,
                    fine=SocialFineHandGesture.PEACE_SIGN,
                    fine_confidence=0.61,
                ),
            )
            replay = lane.observe(
                observed_at=10.6,
                observation=_authoritative_observation(
                    token=3,
                    fine=SocialFineHandGesture.PEACE_SIGN,
                    fine_confidence=0.62,
                ),
            )

        self.assertTrue(emitted.active)
        self.assertFalse(cooldown.active)
        self.assertEqual(cooldown.reason, "live_gesture_cooldown")
        self.assertFalse(still_same.active)
        self.assertEqual(still_same.reason, "live_gesture_already_active")
        self.assertTrue(replay.active)
        self.assertEqual(replay.symbol.value, "victory_hand")

    def test_lane_ignores_unsupported_authoritative_gestures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            wave = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=4,
                    coarse=SocialGestureEvent.WAVE,
                    coarse_confidence=0.91,
                ),
            )
            palm = lane.observe(
                observed_at=10.1,
                observation=_authoritative_observation(
                    token=5,
                    fine=SocialFineHandGesture.OPEN_PALM,
                    fine_confidence=0.98,
                ),
            )

        self.assertFalse(wave.active)
        self.assertEqual(wave.reason, "no_supported_live_gesture")
        self.assertFalse(palm.active)
        self.assertEqual(palm.reason, "no_supported_live_gesture")


if __name__ == "__main__":
    unittest.main()
