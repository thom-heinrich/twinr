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


class SocialTriggerEngineTests(unittest.TestCase):
    def observe(
        self,
        engine: SocialTriggerEngine,
        now: float,
        *,
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


if __name__ == "__main__":
    unittest.main()
