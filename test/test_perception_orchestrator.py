from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.proactive.runtime.perception_orchestrator import PerceptionStreamOrchestrator
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot
from twinr.proactive.social.engine import SocialAudioObservation, SocialFineHandGesture, SocialVisionObservation
from twinr.proactive.social.perception_stream import (
    PerceptionGestureStreamObservation,
    PerceptionStreamObservation,
)


class _CameraSnapshotStub:
    def __init__(self, facts: dict[str, object], *, last_camera_frame_at: float | None = None) -> None:
        self._facts = dict(facts)
        self.last_camera_frame_at = last_camera_frame_at

    def to_automation_facts(self) -> dict[str, object]:
        return dict(self._facts)


class PerceptionStreamOrchestratorTests(unittest.TestCase):
    def test_observe_attention_reuses_precomputed_speaker_association(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PerceptionStreamOrchestrator.from_config(TwinrConfig(project_root=temp_dir))
            speaker_association = ReSpeakerSpeakerAssociationSnapshot(
                observed_at=10.0,
                state="primary_visible_person_associated",
                associated=True,
                target_id="primary_visible_person",
                confidence=0.94,
                camera_person_count=1,
                direction_confidence=0.91,
                azimuth_deg=12,
                primary_person_zone="right",
            )

            snapshot = orchestrator.observe_attention(
                observed_at=10.0,
                source="display_attention_refresh",
                captured_at=9.8,
                camera_snapshot=_CameraSnapshotStub(
                    {
                        "person_visible": True,
                        "person_count": 1,
                        "person_count_unknown": False,
                        "primary_person_zone": "right",
                        "primary_person_center_x": 0.74,
                        "primary_person_center_y": 0.48,
                        "visible_persons": (),
                        "looking_toward_device": True,
                        "person_near_device": True,
                        "engaged_with_device": True,
                        "visual_attention_score": 0.82,
                        "showing_intent_likely": False,
                    },
                    last_camera_frame_at=9.8,
                ),
                audio_observation=SocialAudioObservation(
                    speech_detected=True,
                    azimuth_deg=12,
                    direction_confidence=0.91,
                ),
                audio_policy_snapshot=None,
                runtime_status="waiting",
                presence_session_id=7,
                speaker_association=speaker_association,
            )

        self.assertIsNotNone(snapshot.attention)
        assert snapshot.attention is not None
        self.assertEqual(snapshot.source, "display_attention_refresh")
        self.assertEqual(snapshot.captured_at, 9.8)
        self.assertEqual(snapshot.attention.speaker_association, speaker_association)
        self.assertEqual(snapshot.attention.attention_target.focus_source, "speaker_association")
        self.assertTrue(snapshot.attention.attention_target.speaker_locked)
        self.assertEqual(
            snapshot.attention.live_facts["speaker_association"],
            speaker_association.to_automation_facts(),
        )
        self.assertEqual(
            snapshot.attention.live_facts["attention_target"],
            snapshot.attention.attention_target.to_automation_facts(),
        )

    def test_observe_gesture_drives_ack_and_wakeup_from_same_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = PerceptionStreamOrchestrator.from_config(
                TwinrConfig(
                    project_root=temp_dir,
                    gesture_wakeup_enabled=True,
                    gesture_wakeup_trigger="peace_sign",
                    gesture_wakeup_cooldown_s=3.0,
                )
            )
            observation = SocialVisionObservation(
                hand_or_object_near_camera=True,
                fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN,
                fine_hand_gesture_confidence=0.93,
                perception_stream=PerceptionStreamObservation(
                    gesture=PerceptionGestureStreamObservation(
                        authoritative=True,
                        activation_key="fine:peace_sign",
                        activation_token=42,
                        activation_rising=True,
                    )
                ),
            )

            first = orchestrator.observe_gesture(
                observed_at=11.0,
                source="gesture_stream",
                captured_at=10.9,
                vision_observation=observation,
            )
            held = orchestrator.observe_gesture(
                observed_at=11.2,
                source="gesture_stream",
                captured_at=11.1,
                vision_observation=observation,
            )

        self.assertIsNotNone(first.gesture)
        assert first.gesture is not None
        self.assertTrue(first.gesture.ack_decision.active)
        self.assertTrue(first.gesture.wakeup_decision.active)
        self.assertEqual(first.gesture.wakeup_decision.reason, "gesture_wakeup:peace_sign")
        self.assertIsNotNone(held.gesture)
        assert held.gesture is not None
        self.assertFalse(held.gesture.ack_decision.active)
        self.assertEqual(held.gesture.ack_decision.reason, "live_gesture_already_active")
        self.assertFalse(held.gesture.wakeup_decision.active)
        self.assertEqual(held.gesture.wakeup_decision.reason, "gesture_wakeup_already_active")


if __name__ == "__main__":
    unittest.main()
