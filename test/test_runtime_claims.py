from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.runtime.affect_proxy import derive_affect_proxy
from twinr.proactive.runtime.ambiguous_room_guard import derive_ambiguous_room_guard
from twinr.proactive.runtime.known_user_hint import derive_known_user_hint


class AmbiguousRoomGuardTests(unittest.TestCase):
    def test_blocks_multi_person_context(self) -> None:
        snapshot = derive_ambiguous_room_guard(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "person_count_unknown": False,
                },
                "audio_policy": {},
            },
        )

        self.assertTrue(snapshot.guard_active)
        self.assertEqual(snapshot.reason, "multi_person_context")
        self.assertEqual(snapshot.policy_recommendation, "block_targeted_inference")
        self.assertGreater(snapshot.claim.confidence, 0.9)

    def test_clears_single_visible_person_quiet_room(self) -> None:
        snapshot = derive_ambiguous_room_guard(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "person_count_unknown": False,
                },
                "audio_policy": {
                    "presence_audio_active": False,
                    "recent_follow_up_speech": False,
                    "resume_window_open": False,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
        )

        self.assertFalse(snapshot.guard_active)
        self.assertTrue(snapshot.clear)
        self.assertIsNone(snapshot.reason)
        self.assertEqual(snapshot.policy_recommendation, "clear")
        self.assertGreater(snapshot.claim.confidence, 0.8)


class KnownUserHintTests(unittest.TestCase):
    def test_likely_user_requires_clear_room_context(self) -> None:
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=20)).isoformat().replace("+00:00", "Z")
        snapshot = derive_known_user_hint(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "person_count_unknown": False,
                },
                "audio_policy": {
                    "presence_audio_active": False,
                    "recent_follow_up_speech": False,
                    "resume_window_open": False,
                },
                "vad": {
                    "speech_detected": False,
                },
            },
            voice_status="likely_user",
            voice_confidence=0.83,
            voice_checked_at=checked_at,
            now_utc=now_utc,
        )

        self.assertEqual(snapshot.state, "likely_main_user")
        self.assertTrue(snapshot.matches_main_user)
        self.assertEqual(snapshot.policy_recommendation, "calm_personalization_only")
        self.assertTrue(snapshot.claim.requires_confirmation)
        self.assertGreater(snapshot.claim.confidence, 0.8)

    def test_ambiguous_room_blocks_known_user_hint(self) -> None:
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=10)).isoformat().replace("+00:00", "Z")
        snapshot = derive_known_user_hint(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "person_count_unknown": False,
                },
                "audio_policy": {},
            },
            voice_status="likely_user",
            voice_confidence=0.9,
            voice_checked_at=checked_at,
            now_utc=now_utc,
        )

        self.assertEqual(snapshot.state, "blocked_ambiguous_room")
        self.assertFalse(snapshot.matches_main_user)
        self.assertEqual(snapshot.block_reason, "multi_person_context")
        self.assertEqual(snapshot.policy_recommendation, "blocked")


class AffectProxyTests(unittest.TestCase):
    def test_slumped_quiet_low_motion_becomes_concern_cue(self) -> None:
        snapshot = derive_affect_proxy(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 1,
                    "person_count_unknown": False,
                    "body_pose": "slumped",
                    "body_pose_unknown": False,
                    "looking_toward_device": False,
                    "looking_toward_device_unknown": False,
                    "engaged_with_device": False,
                    "engaged_with_device_unknown": False,
                },
                "vad": {
                    "room_quiet": True,
                },
                "pir": {
                    "low_motion": True,
                },
                "audio_policy": {
                    "presence_audio_active": False,
                    "recent_follow_up_speech": False,
                    "resume_window_open": False,
                },
            },
        )

        self.assertEqual(snapshot.state, "concern_cue")
        self.assertEqual(snapshot.policy_recommendation, "prompt_only")
        self.assertTrue(snapshot.claim.requires_confirmation)
        self.assertGreater(snapshot.claim.confidence, 0.7)

    def test_ambiguous_room_suppresses_affect_proxy(self) -> None:
        snapshot = derive_affect_proxy(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "person_count_unknown": False,
                    "body_pose": "upright",
                    "body_pose_unknown": False,
                    "smiling": True,
                    "smiling_unknown": False,
                    "looking_toward_device": True,
                    "looking_toward_device_unknown": False,
                },
                "audio_policy": {},
            },
        )

        self.assertEqual(snapshot.state, "unknown")
        self.assertEqual(snapshot.block_reason, "multi_person_context")
        self.assertEqual(snapshot.policy_recommendation, "ignore")


if __name__ == "__main__":
    unittest.main()
