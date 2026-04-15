from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
    resolved_source: str | None = "live_stream",
    hand_or_object_near_camera: bool = True,
) -> SocialVisionObservation:
    return SocialVisionObservation(
        hand_or_object_near_camera=hand_or_object_near_camera,
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
                hand_or_object_near_camera=hand_or_object_near_camera,
                resolved_source=resolved_source,
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
                    fine_confidence=0.68,
                ),
            )
            held = lane.observe(
                observed_at=10.2,
                observation=_authoritative_observation(
                    token=1,
                    fine=SocialFineHandGesture.THUMBS_UP,
                    fine_confidence=0.69,
                ),
            )

        self.assertTrue(first.active)
        self.assertEqual(first.symbol.value, "thumbs_up")
        self.assertFalse(held.active)
        self.assertEqual(held.reason, "live_gesture_already_active")

    def test_lane_uses_conservative_runtime_calibration_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            thumbs_up = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=1,
                    fine=SocialFineHandGesture.THUMBS_UP,
                    fine_confidence=0.68,
                ),
            )
            thumbs_down = lane.observe(
                observed_at=11.0,
                observation=_authoritative_observation(
                    token=2,
                    fine=SocialFineHandGesture.THUMBS_DOWN,
                    fine_confidence=0.78,
                ),
            )

        self.assertTrue(thumbs_up.active)
        self.assertEqual(thumbs_up.symbol.value, "thumbs_up")
        self.assertTrue(thumbs_down.active)
        self.assertEqual(thumbs_down.symbol.value, "thumbs_down")

    def test_lane_blocks_weak_thumbs_below_runtime_calibration_floor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            weak_thumbs_up = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=1,
                    fine=SocialFineHandGesture.THUMBS_UP,
                    fine_confidence=0.67,
                ),
            )
            weak_thumbs_down = lane.observe(
                observed_at=11.0,
                observation=_authoritative_observation(
                    token=2,
                    fine=SocialFineHandGesture.THUMBS_DOWN,
                    fine_confidence=0.56,
                ),
            )

        self.assertFalse(weak_thumbs_up.active)
        self.assertEqual(weak_thumbs_up.reason, "no_supported_live_gesture")
        self.assertFalse(weak_thumbs_down.active)
        self.assertEqual(weak_thumbs_down.reason, "no_supported_live_gesture")

    def test_lane_explicit_global_override_can_lower_supported_fine_gesture_floor(self) -> None:
        lane = GestureAckLane.from_config(
            {
                "gesture_ack_lane": {
                    "min_fine_confidence": 0.44,
                }
            }
        )

        thumbs_up = lane.observe(
            observed_at=10.0,
            observation=_authoritative_observation(
                token=1,
                fine=SocialFineHandGesture.THUMBS_UP,
                fine_confidence=0.48,
            ),
        )
        thumbs_down = lane.observe(
            observed_at=11.0,
            observation=_authoritative_observation(
                token=2,
                fine=SocialFineHandGesture.THUMBS_DOWN,
                fine_confidence=0.446,
            ),
        )

        self.assertTrue(thumbs_up.active)
        self.assertEqual(thumbs_up.symbol.value, "thumbs_up")
        self.assertTrue(thumbs_down.active)
        self.assertEqual(thumbs_down.symbol.value, "thumbs_down")

    def test_lane_applies_repeat_hold_across_new_tokens_for_same_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))
            with patch(
                "twinr.proactive.runtime.gesture_ack_lane.time.monotonic",
                side_effect=[10.0, 10.2, 10.3, 10.6],
            ):
                emitted = lane.observe(
                    observed_at=10.0,
                    observation=_authoritative_observation(
                        token=1,
                        fine=SocialFineHandGesture.PEACE_SIGN,
                        fine_confidence=0.78,
                    ),
                )
                cooldown = lane.observe(
                    observed_at=10.2,
                    observation=_authoritative_observation(
                        token=2,
                        fine=SocialFineHandGesture.PEACE_SIGN,
                        fine_confidence=0.79,
                    ),
                )
                still_same = lane.observe(
                    observed_at=10.3,
                    observation=_authoritative_observation(
                        token=2,
                        fine=SocialFineHandGesture.PEACE_SIGN,
                        fine_confidence=0.79,
                    ),
                )
                replay = lane.observe(
                    observed_at=10.6,
                    observation=_authoritative_observation(
                        token=3,
                        fine=SocialFineHandGesture.PEACE_SIGN,
                        fine_confidence=0.80,
                    ),
                )

        self.assertTrue(emitted.active)
        self.assertFalse(cooldown.active)
        self.assertEqual(cooldown.reason, "live_gesture_cooldown")
        self.assertFalse(still_same.active)
        self.assertEqual(still_same.reason, "live_gesture_cooldown")
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

    def test_lane_rejects_supported_gesture_without_current_hand_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=6,
                    fine=SocialFineHandGesture.THUMBS_DOWN,
                    fine_confidence=0.88,
                    resolved_source="person_roi",
                    hand_or_object_near_camera=False,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "live_gesture_missing_hand_evidence")

    def test_lane_rejects_supported_gesture_from_stale_rescue_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lane = GestureAckLane.from_config(TwinrConfig(project_root=temp_dir))

            decision = lane.observe(
                observed_at=10.0,
                observation=_authoritative_observation(
                    token=7,
                    fine=SocialFineHandGesture.THUMBS_DOWN,
                    fine_confidence=0.88,
                    resolved_source="recent_person_roi",
                    hand_or_object_near_camera=True,
                ),
            )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "live_gesture_unsafe_source")


if __name__ == "__main__":
    unittest.main()
