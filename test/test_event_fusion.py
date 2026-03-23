from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.event_fusion import (
    AudioClassifierHints,
    EventFusionPolicyContext,
    FusionActionLevel,
    MultimodalEventFusionConfig,
    MultimodalEventFusionTracker,
    RollingWindowBuffer,
    build_fused_claim,
    derive_audio_micro_events,
    derive_vision_sequences,
)
from twinr.proactive.social.engine import (
    SocialAudioObservation,
    SocialBodyPose,
    SocialMotionState,
    SocialObservation,
    SocialVisionObservation,
)


class RollingWindowBufferTests(unittest.TestCase):
    def test_prunes_samples_older_than_horizon(self) -> None:
        buffer = RollingWindowBuffer[int](horizon_s=4.0)

        buffer.append(1.0, 1)
        buffer.append(3.0, 2)
        buffer.append(7.0, 3)

        self.assertEqual([sample.value for sample in buffer.snapshot()], [2, 3])
        self.assertEqual(buffer.oldest_timestamp, 3.0)
        self.assertEqual(buffer.newest_timestamp, 7.0)

    def test_clamps_regressing_timestamp_to_keep_monotonic_order(self) -> None:
        buffer = RollingWindowBuffer[str](horizon_s=5.0)

        first = buffer.append(5.0, "first")
        second = buffer.append(4.0, "second")

        self.assertEqual(first.observed_at, 5.0)
        self.assertEqual(second.observed_at, 5.0)


class AudioEventTests(unittest.TestCase):
    def test_derives_media_and_hint_backed_audio_events(self) -> None:
        events = derive_audio_micro_events(
            observed_at=12.0,
            observation=SocialAudioObservation(
                speech_detected=True,
                background_media_likely=True,
                speech_overlap_likely=True,
            ),
            hints=AudioClassifierHints(laugh_like_confidence=0.82),
        )

        self.assertEqual(
            [event.kind.value for event in events],
            [
                "speech_activity",
                "background_media_likely",
                "speech_overlap_likely",
                "laugh_like_audio",
            ],
        )


class VisionSequenceTests(unittest.TestCase):
    def test_derives_drop_and_floor_sequences_from_recent_pose_history(self) -> None:
        buffer = RollingWindowBuffer[SocialVisionObservation](horizon_s=8.0)
        buffer.append(
            1.0,
            SocialVisionObservation(
                person_visible=True,
                person_count=1,
                body_pose=SocialBodyPose.SEATED,
                motion_state=SocialMotionState.STILL,
            ),
        )
        buffer.append(
            2.0,
            SocialVisionObservation(
                person_visible=True,
                person_count=1,
                body_pose=SocialBodyPose.FLOOR,
                motion_state=SocialMotionState.STILL,
            ),
        )
        buffer.append(
            5.5,
            SocialVisionObservation(
                person_visible=True,
                person_count=1,
                body_pose=SocialBodyPose.FLOOR,
                motion_state=SocialMotionState.STILL,
            ),
        )

        sequences = derive_vision_sequences(observation_buffer=buffer, now=5.5)

        self.assertEqual(
            [sequence.kind.value for sequence in sequences],
            [
                "floor_pose_entered",
                "downward_transition",
                "floor_stillness",
            ],
        )


class FusedClaimTests(unittest.TestCase):
    def test_claim_builder_blocks_delivery_when_media_and_multi_person_are_active(self) -> None:
        claim = build_fused_claim(
            state="possible_fall",
            confidence=0.84,
            source="vision_downward_transition_plus_floor_pose_sequence",
            policy_context=EventFusionPolicyContext(
                background_media_likely=True,
                room_busy_or_overlapping=False,
                multi_person_context=True,
            ),
            window_start_s=10.0,
            window_end_s=15.0,
            preferred_action_level=FusionActionLevel.PROMPT_ONLY,
            supporting_vision_events=("downward_transition", "floor_pose_entered"),
        )

        self.assertFalse(claim.delivery_allowed)
        self.assertEqual(claim.action_level, FusionActionLevel.IGNORE)
        self.assertTrue(claim.requires_confirmation)
        self.assertEqual(claim.blocked_by, ("background_media_active", "multi_person_context"))


class FusionTrackerTests(unittest.TestCase):
    def test_emits_possible_fall_and_floor_stillness_after_drop(self) -> None:
        tracker = MultimodalEventFusionTracker()

        tracker.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(),
            )
        )
        tracker.observe(
            SocialObservation(
                observed_at=2.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.FLOOR,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(),
            )
        )
        claims = tracker.observe(
            SocialObservation(
                observed_at=5.5,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.FLOOR,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(),
            )
        )

        self.assertEqual([claim.state for claim in claims], ["possible_fall", "floor_stillness_after_drop"])
        self.assertTrue(all(claim.requires_confirmation for claim in claims))
        self.assertTrue(all(claim.delivery_allowed for claim in claims))
        possible_fall = claims[0]
        self.assertTrue(possible_fall.review_recommended)
        self.assertIsNotNone(possible_fall.keyframe_review_plan)
        self.assertEqual(
            [frame.role for frame in possible_fall.keyframe_review_plan.frames],
            ["onset", "peak", "latest"],
        )

    def test_blocks_person_targeted_fused_claims_in_multi_person_context(self) -> None:
        tracker = MultimodalEventFusionTracker()

        tracker.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=2,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(),
            ),
            audio_hints=AudioClassifierHints(cry_like_confidence=0.81),
        )
        claims = tracker.observe(
            SocialObservation(
                observed_at=4.5,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=2,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(),
            ),
            audio_hints=AudioClassifierHints(cry_like_confidence=0.83),
        )

        states = [claim.state for claim in claims]
        self.assertIn("distress_possible", states)
        self.assertIn("cry_like_distress_possible", states)
        distress_claim = next(claim for claim in claims if claim.state == "distress_possible")
        cry_claim = next(claim for claim in claims if claim.state == "cry_like_distress_possible")
        self.assertFalse(distress_claim.delivery_allowed)
        self.assertFalse(cry_claim.delivery_allowed)
        self.assertIn("multi_person_context", distress_claim.blocked_by)
        self.assertIn("multi_person_context", cry_claim.blocked_by)
        self.assertTrue(cry_claim.review_recommended)
        self.assertIsNotNone(cry_claim.keyframe_review_plan)

    def test_blocks_shout_like_audio_when_background_media_is_active(self) -> None:
        tracker = MultimodalEventFusionTracker()

        claims = tracker.observe(
            SocialObservation(
                observed_at=3.0,
                inspected=False,
                audio=SocialAudioObservation(
                    speech_detected=True,
                    distress_detected=True,
                    background_media_likely=True,
                ),
            )
        )

        self.assertEqual([claim.state for claim in claims], ["shout_like_audio"])
        self.assertFalse(claims[0].delivery_allowed)
        self.assertIn("background_media_active", claims[0].blocked_by)
        self.assertFalse(claims[0].review_recommended)
        self.assertIsNone(claims[0].keyframe_review_plan)

    def test_emits_distress_possible_from_distress_audio_and_slumped_sequence(self) -> None:
        tracker = MultimodalEventFusionTracker()

        tracker.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(speech_detected=False),
            )
        )
        claims = tracker.observe(
            SocialObservation(
                observed_at=4.5,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.SLUMPED,
                    motion_state=SocialMotionState.STILL,
                ),
                audio=SocialAudioObservation(
                    speech_detected=True,
                    distress_detected=True,
                ),
            )
        )

        states = [claim.state for claim in claims]
        self.assertIn("shout_like_audio", states)
        self.assertIn("distress_possible", states)
        distress_claim = next(claim for claim in claims if claim.state == "distress_possible")
        self.assertTrue(distress_claim.delivery_allowed)
        self.assertEqual(distress_claim.action_level, FusionActionLevel.REVIEW_ONLY)
        self.assertTrue(distress_claim.review_recommended)
        self.assertIn("shout_like_audio", distress_claim.supporting_audio_events)
        self.assertIn("slumped_quiet", distress_claim.supporting_vision_events)

    def test_emits_laugh_like_positive_contact_with_single_person_context(self) -> None:
        tracker = MultimodalEventFusionTracker()

        claims = tracker.observe(
            SocialObservation(
                observed_at=2.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.87),
        )
        claims = tracker.observe(
            SocialObservation(
                observed_at=4.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.9),
        )

        self.assertEqual([claim.state for claim in claims], ["laugh_like_positive_contact"])
        self.assertTrue(claims[0].delivery_allowed)
        self.assertEqual(claims[0].action_level, FusionActionLevel.PROMPT_ONLY)
        self.assertFalse(claims[0].review_recommended)

    def test_temporal_decay_reduces_stale_single_event_confidence(self) -> None:
        tracker = MultimodalEventFusionTracker(
            MultimodalEventFusionConfig(
                temporal_decay_half_life_audio_s=0.75,
                multi_scale_windows_s=(1.0, 2.0, 4.0),
            )
        )

        recent_claim = tracker.observe(
            SocialObservation(
                observed_at=1.0,
                inspected=False,
                audio=SocialAudioObservation(
                    speech_detected=True,
                    distress_detected=True,
                ),
            )
        )[0]
        stale_claim = tracker.observe(
            SocialObservation(
                observed_at=4.0,
                inspected=False,
                audio=SocialAudioObservation(),
            )
        )[0]

        self.assertEqual(recent_claim.state, "shout_like_audio")
        self.assertEqual(stale_claim.state, "shout_like_audio")
        self.assertLess(stale_claim.confidence, recent_claim.confidence)

    def test_sequence_fusion_uses_repeated_support_not_only_latest_match(self) -> None:
        sparse_tracker = MultimodalEventFusionTracker()
        dense_tracker = MultimodalEventFusionTracker()

        sparse_tracker.observe(
            SocialObservation(
                observed_at=2.5,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.88),
        )
        sparse_claim = sparse_tracker.observe(
            SocialObservation(
                observed_at=4.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.88),
        )[0]

        dense_tracker.observe(
            SocialObservation(
                observed_at=2.4,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.84),
        )
        dense_tracker.observe(
            SocialObservation(
                observed_at=3.2,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.86),
        )
        dense_claim = dense_tracker.observe(
            SocialObservation(
                observed_at=4.0,
                inspected=True,
                vision=SocialVisionObservation(
                    person_visible=True,
                    person_count=1,
                    body_pose=SocialBodyPose.UPRIGHT,
                    motion_state=SocialMotionState.STILL,
                    smiling=True,
                    looking_toward_device=True,
                    engaged_with_device=True,
                ),
                audio=SocialAudioObservation(speech_detected=True),
            ),
            audio_hints=AudioClassifierHints(laugh_like_confidence=0.88),
        )[0]

        self.assertEqual(sparse_claim.state, "laugh_like_positive_contact")
        self.assertEqual(dense_claim.state, "laugh_like_positive_contact")
        self.assertGreater(dense_claim.confidence, sparse_claim.confidence)


if __name__ == "__main__":
    unittest.main()
