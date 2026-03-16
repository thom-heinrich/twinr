from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive import (
    SocialAudioObservation,
    SocialBodyPose,
    SocialObservation,
    SocialTriggerEngine,
    SocialVisionObservation,
)
from twinr.config import TwinrConfig


class SocialTriggerEngineTests(unittest.TestCase):
    def possible_fall_evaluation(self, engine: SocialTriggerEngine):
        return next(item for item in engine.last_evaluations if item.trigger_id == "possible_fall")

    def observe(
        self,
        engine: SocialTriggerEngine,
        now: float,
        *,
        inspected: bool = True,
        pir: bool = False,
        low_motion: bool = False,
        person_visible: bool = False,
        looking: bool = False,
        pose: SocialBodyPose = SocialBodyPose.UNKNOWN,
        smiling: bool = False,
        showing: bool = False,
        speech: bool = False,
        distress: bool = False,
    ):
        return engine.observe(
            SocialObservation(
                observed_at=now,
                inspected=inspected,
                pir_motion_detected=pir,
                low_motion=low_motion,
                vision=SocialVisionObservation(
                    person_visible=person_visible,
                    looking_toward_device=looking,
                    body_pose=pose,
                    smiling=smiling,
                    hand_or_object_near_camera=showing,
                ),
                audio=SocialAudioObservation(
                    speech_detected=speech,
                    distress_detected=distress,
                ),
            )
        )

    def test_person_returned_requires_long_absence_and_recent_pir(self) -> None:
        engine = SocialTriggerEngine(user_name="Thom")
        self.observe(engine, 0.0, person_visible=False)
        decision = self.observe(
            engine,
            21.0 * 60.0,
            pir=True,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "person_returned")
        self.assertEqual(decision.prompt, "Hey Thom, schön dich zu sehen. Wie geht's dir?")
        self.assertGreaterEqual(decision.score, decision.threshold)
        self.assertTrue(decision.evidence)

    def test_attention_window_triggers_after_quiet_gaze_hold(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 10.0, person_visible=True, looking=True, pose=SocialBodyPose.UPRIGHT, speech=False)
        decision = self.observe(
            engine,
            16.2,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            speech=False,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "attention_window")
        self.assertGreaterEqual(decision.score, decision.threshold)
        self.assertTrue(any(item.key == "looking_hold" for item in decision.evidence))

    def test_attention_window_does_not_trigger_when_speech_interrupts_quiet_hold(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 10.0, person_visible=True, looking=True, pose=SocialBodyPose.UPRIGHT, speech=False)
        decision = self.observe(
            engine,
            16.2,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            speech=True,
        )

        self.assertIsNone(decision)
        evaluation = next(item for item in engine.last_evaluations if item.trigger_id == "attention_window")
        self.assertLess(evaluation.score, evaluation.threshold)

    def test_slumped_quiet_requires_slumped_low_motion_and_quiet(self) -> None:
        engine = SocialTriggerEngine(user_name="Thom")
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            20.1,
            person_visible=True,
            pose=SocialBodyPose.SLUMPED,
            low_motion=True,
            speech=False,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "slumped_quiet")
        self.assertEqual(decision.prompt, "Hey Thom, ist alles in Ordnung?")
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_possible_fall_triggers_after_floor_transition_and_stillness(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=False)
        self.observe(engine, 1.0, person_visible=True, pose=SocialBodyPose.FLOOR, low_motion=True)
        decision = self.observe(
            engine,
            11.5,
            person_visible=True,
            pose=SocialBodyPose.FLOOR,
            low_motion=True,
            speech=False,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "possible_fall")
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_floor_stillness_beats_possible_fall_when_quiet_longer(self) -> None:
        engine = SocialTriggerEngine(user_name="Thom")
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.UPRIGHT)
        self.observe(engine, 1.0, person_visible=True, pose=SocialBodyPose.FLOOR, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            21.5,
            person_visible=True,
            pose=SocialBodyPose.FLOOR,
            low_motion=True,
            speech=False,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "floor_stillness")
        self.assertEqual(decision.prompt, "Hey Thom, antworte mir kurz: Ist alles okay?")
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_possible_fall_triggers_when_slumped_person_drops_out_of_view_and_stays_still(self) -> None:
        config = TwinrConfig(
            proactive_possible_fall_visibility_loss_arming_s=2.0,
            proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
            proactive_possible_fall_visibility_loss_hold_s=10.0,
        )
        engine = SocialTriggerEngine.from_config(config)
        self.observe(engine, 0.0, pir=True, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=False, speech=False)
        self.observe(engine, 2.5, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=True, speech=False)
        self.observe(engine, 3.0, person_visible=False, pose=SocialBodyPose.UNKNOWN, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            13.5,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "possible_fall")
        self.assertTrue(any(item.key == "fall_transition_signal" for item in decision.evidence))
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_possible_fall_does_not_trigger_when_person_briefly_looks_slumped_before_sitting_out_of_view(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=False, speech=False)
        self.observe(engine, 6.5, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=True, speech=False)
        self.observe(engine, 7.0, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=True, speech=False)
        self.observe(engine, 8.0, person_visible=False, pose=SocialBodyPose.UNKNOWN, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            11.5,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNone(decision)
        possible_fall = self.possible_fall_evaluation(engine)
        self.assertLess(possible_fall.score, possible_fall.threshold)

    def test_possible_fall_does_not_trigger_when_upright_person_just_leaves_frame(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=False, speech=False)
        self.observe(engine, 2.5, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=True, speech=False)
        self.observe(engine, 3.0, person_visible=False, pose=SocialBodyPose.UNKNOWN, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            13.5,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNone(decision)
        possible_fall = self.possible_fall_evaluation(engine)
        self.assertLess(possible_fall.score, possible_fall.threshold)

    def test_possible_fall_does_not_borrow_quiet_or_stillness_from_before_visibility_loss(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=True, speech=False)
        self.observe(engine, 30.0, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            30.5,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNone(decision)
        possible_fall = self.possible_fall_evaluation(engine)
        self.assertLess(possible_fall.score, possible_fall.threshold)

    def test_possible_fall_does_not_trigger_immediately_on_visibility_loss(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=False, speech=False)
        self.observe(engine, 2.5, person_visible=True, pose=SocialBodyPose.UPRIGHT, low_motion=True, speech=False)
        self.observe(engine, 3.0, person_visible=False, pose=SocialBodyPose.UNKNOWN, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            4.0,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=True,
        )

        self.assertIsNone(decision)
        possible_fall = self.possible_fall_evaluation(engine)
        self.assertLess(possible_fall.score, possible_fall.threshold)

    def test_possible_fall_visibility_loss_path_requires_longer_missing_hold_than_floor_path(self) -> None:
        config = TwinrConfig(
            proactive_possible_fall_stillness_s=4.0,
            proactive_possible_fall_visibility_loss_hold_s=8.0,
            proactive_possible_fall_visibility_loss_arming_s=2.0,
            proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
            proactive_possible_fall_score_threshold=0.65,
        )
        engine = SocialTriggerEngine.from_config(config)

        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=False, speech=False)
        self.observe(engine, 4.0, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=True, speech=False)
        self.observe(engine, 5.0, person_visible=False, pose=SocialBodyPose.UNKNOWN, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            8.4,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNone(decision)
        possible_fall = self.possible_fall_evaluation(engine)
        self.assertEqual(possible_fall.blocked_reason, "visibility_loss_hold_incomplete")

    def test_possible_fall_visibility_loss_path_can_still_trigger_after_longer_missing_hold(self) -> None:
        config = TwinrConfig(
            proactive_possible_fall_stillness_s=4.0,
            proactive_possible_fall_visibility_loss_hold_s=8.0,
            proactive_possible_fall_visibility_loss_arming_s=2.0,
            proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
            proactive_possible_fall_score_threshold=0.65,
        )
        engine = SocialTriggerEngine.from_config(config)

        self.observe(engine, 0.0, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=False, speech=False)
        self.observe(engine, 4.0, person_visible=True, pose=SocialBodyPose.SLUMPED, low_motion=True, speech=False)
        self.observe(engine, 5.0, person_visible=False, pose=SocialBodyPose.UNKNOWN, low_motion=True, speech=False)
        decision = self.observe(
            engine,
            13.6,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "possible_fall")

    def test_possible_fall_visibility_loss_path_requires_confirmed_visible_person_across_multiple_inspections(self) -> None:
        config = TwinrConfig(
            proactive_possible_fall_stillness_s=4.0,
            proactive_possible_fall_visibility_loss_hold_s=8.0,
            proactive_possible_fall_visibility_loss_arming_s=2.0,
            proactive_possible_fall_slumped_visibility_loss_arming_s=2.0,
            proactive_possible_fall_score_threshold=0.65,
        )
        engine = SocialTriggerEngine.from_config(config)

        self.observe(
            engine,
            0.0,
            pir=True,
            person_visible=True,
            pose=SocialBodyPose.SLUMPED,
            low_motion=False,
            speech=False,
        )
        decision = self.observe(
            engine,
            10.5,
            person_visible=False,
            pose=SocialBodyPose.UNKNOWN,
            low_motion=True,
            speech=False,
        )

        self.assertIsNone(decision)
        possible_fall = self.possible_fall_evaluation(engine)
        self.assertLess(possible_fall.score, possible_fall.threshold)

    def test_showing_intent_needs_object_near_camera(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(
            engine,
            5.0,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            showing=True,
        )
        decision = self.observe(
            engine,
            6.7,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            showing=True,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "showing_intent")
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_distress_possible_requires_distress_audio_and_visible_person(self) -> None:
        engine = SocialTriggerEngine(user_name="Thom")
        self.observe(engine, 3.0, person_visible=True, pose=SocialBodyPose.SLUMPED, distress=True)
        decision = self.observe(
            engine,
            6.5,
            person_visible=True,
            pose=SocialBodyPose.SLUMPED,
            distress=True,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "distress_possible")
        self.assertEqual(decision.prompt, "Hey Thom, ich wollte nur kurz fragen, ob alles in Ordnung ist.")
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_positive_contact_uses_smile_after_short_hold(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 7.0, person_visible=True, looking=True, pose=SocialBodyPose.UPRIGHT, smiling=True)
        decision = self.observe(
            engine,
            8.7,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            smiling=True,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.trigger_id, "positive_contact")
        self.assertGreaterEqual(decision.score, decision.threshold)

    def test_cooldown_blocks_duplicate_attention_window(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=True, looking=True, pose=SocialBodyPose.UPRIGHT, speech=False)
        first = self.observe(
            engine,
            6.2,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            speech=False,
        )
        second = self.observe(
            engine,
            8.0,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
            speech=False,
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_best_evaluation_keeps_near_miss_score_visible(self) -> None:
        engine = SocialTriggerEngine()
        self.observe(engine, 0.0, person_visible=False)
        decision = self.observe(
            engine,
            21.0 * 60.0,
            person_visible=True,
            looking=True,
            pose=SocialBodyPose.UPRIGHT,
        )

        self.assertIsNone(decision)
        self.assertIsNotNone(engine.best_evaluation)
        self.assertEqual(engine.best_evaluation.trigger_id, "person_returned")
        self.assertLess(engine.best_evaluation.score, engine.best_evaluation.threshold)
        self.assertIsNone(engine.best_evaluation.blocked_reason)

    def test_engine_from_config_uses_proactive_thresholds(self) -> None:
        config = TwinrConfig(
            user_display_name="Thom",
            proactive_attention_window_s=9.0,
            proactive_showing_intent_score_threshold=0.73,
        )

        engine = SocialTriggerEngine.from_config(config)

        self.assertEqual(engine.user_name, "Thom")
        self.assertEqual(engine.thresholds.attention_window_s, 9.0)
        self.assertEqual(engine.thresholds.showing_intent_score_threshold, 0.73)


if __name__ == "__main__":
    unittest.main()
