from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.person_state import derive_person_state


class PersonStateTests(unittest.TestCase):
    def test_person_state_aggregates_clear_engaged_single_person(self) -> None:
        snapshot = derive_person_state(
            observed_at=12.0,
            live_facts={
                "sensor": {
                    "observed_at": 12.0,
                    "wakeword_armed": True,
                },
                "pir": {
                    "motion_detected": True,
                    "low_motion": False,
                },
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "looking_toward_device": True,
                    "engaged_with_device": True,
                    "person_near_device": True,
                    "visual_attention_score": 0.91,
                    "showing_intent_likely": True,
                    "hand_or_object_near_camera": True,
                    "fine_hand_gesture": "peace_sign",
                    "fine_hand_gesture_confidence": 0.93,
                },
                "vad": {
                    "speech_detected": True,
                    "room_quiet": False,
                },
                "audio_policy": {
                    "presence_audio_active": True,
                    "recent_follow_up_speech": False,
                    "quiet_window_open": False,
                    "room_busy_or_overlapping": False,
                },
                "speaker_association": {
                    "associated": True,
                    "state": "primary_visible_person_associated",
                    "confidence": 0.88,
                },
                "multimodal_initiative": {
                    "ready": True,
                    "confidence": 0.86,
                    "recommended_channel": "speech",
                },
                "ambiguous_room_guard": {
                    "clear": True,
                    "guard_active": False,
                    "policy_recommendation": "clear",
                    "confidence": 0.9,
                    "source": "camera_plus_audio_policy",
                },
                "known_user_hint": {
                    "state": "likely_main_user_multimodal",
                    "matches_main_user": True,
                    "policy_recommendation": "calm_personalization_only",
                    "confidence": 0.84,
                    "source": "voice_profile_plus_portrait_match_plus_single_visible_person_context",
                    "requires_confirmation": True,
                },
            },
        )

        facts = snapshot.to_automation_facts()
        self.assertEqual(facts["presence_state"]["state"], "occupied_visible")
        self.assertEqual(facts["attention_state"]["state"], "engaged_with_device")
        self.assertEqual(facts["interaction_intent_state"]["state"], "explicit_gesture_request")
        self.assertEqual(facts["conversation_state"]["state"], "device_directed_speech")
        self.assertEqual(facts["identity_state"]["state"], "likely_main_user")
        self.assertTrue(facts["presence_active"])
        self.assertTrue(facts["interaction_ready"])
        self.assertTrue(facts["calm_personalization_allowed"])
        self.assertFalse(facts["targeted_inference_blocked"])
        self.assertEqual(facts["recommended_channel"], "speech")

    def test_person_state_marks_concern_cues_as_risk_not_diagnosis(self) -> None:
        snapshot = derive_person_state(
            observed_at=18.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "body_pose": "slumped",
                },
                "pir": {
                    "motion_detected": False,
                    "low_motion": True,
                },
                "vad": {
                    "speech_detected": False,
                    "room_quiet": True,
                },
                "ambiguous_room_guard": {
                    "clear": True,
                    "guard_active": False,
                    "policy_recommendation": "clear",
                    "confidence": 0.82,
                    "source": "camera_plus_audio_policy",
                },
                "affect_proxy": {
                    "state": "concern_cue",
                    "policy_recommendation": "prompt_only",
                    "confidence": 0.72,
                    "source": "camera_pose_plus_pir_plus_audio",
                    "requires_confirmation": True,
                    "body_pose": "slumped",
                    "room_quiet": True,
                    "low_motion": True,
                },
            },
        )

        facts = snapshot.to_automation_facts()
        self.assertEqual(facts["safety_state"]["kind"], "risk_cue")
        self.assertEqual(facts["safety_state"]["state"], "concern_cue")
        self.assertTrue(facts["safety_state"]["requires_confirmation"])
        self.assertTrue(facts["safety_concern_active"])
        self.assertEqual(facts["recommended_channel"], "speech")

    def test_person_state_blocks_identity_when_room_is_ambiguous(self) -> None:
        snapshot = derive_person_state(
            observed_at=9.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                },
                "ambiguous_room_guard": {
                    "clear": False,
                    "guard_active": True,
                    "reason": "multi_person_context",
                    "policy_recommendation": "block_targeted_inference",
                    "confidence": 0.97,
                    "source": "camera_plus_audio_policy",
                },
                "known_user_hint": {
                    "state": "blocked_ambiguous_room",
                    "matches_main_user": False,
                    "policy_recommendation": "blocked",
                    "block_reason": "multi_person_context",
                    "confidence": 0.97,
                    "source": "voice_profile_plus_ambiguous_room_guard",
                    "requires_confirmation": True,
                },
            },
        )

        facts = snapshot.to_automation_facts()
        self.assertEqual(facts["room_clarity_state"]["state"], "multi_person_context")
        self.assertEqual(facts["identity_state"]["state"], "blocked_ambiguous_room")
        self.assertTrue(facts["targeted_inference_blocked"])
        self.assertFalse(facts["calm_personalization_allowed"])

    def test_person_state_uses_smart_home_only_as_context(self) -> None:
        snapshot = derive_person_state(
            observed_at=30.0,
            live_facts={
                "room_context": {
                    "available": True,
                    "same_room_motion_recent": True,
                    "same_room_button_recent": False,
                    "confidence": 0.81,
                    "source": "smart_home_room_context",
                },
                "home_context": {
                    "available": True,
                    "home_occupied_likely": True,
                    "device_offline": True,
                    "alarm_active": False,
                    "confidence": 0.77,
                    "source": "smart_home_home_context",
                },
            },
        )

        facts = snapshot.to_automation_facts()
        self.assertEqual(facts["presence_state"]["state"], "empty")
        self.assertFalse(facts["presence_active"])
        self.assertEqual(facts["home_context_state"]["state"], "device_offline")
        self.assertTrue(facts["home_occupied_likely"])


if __name__ == "__main__":
    unittest.main()
