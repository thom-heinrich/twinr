from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent.config import TwinrConfig
from twinr.hardware.portrait_match import PortraitMatchObservation
from twinr.proactive.runtime.affect_proxy import derive_affect_proxy
from twinr.proactive.runtime.ambiguous_room_guard import derive_ambiguous_room_guard
from twinr.proactive.runtime.identity_fusion import TemporalIdentityFusionTracker
from twinr.proactive.runtime.known_user_hint import derive_known_user_hint
from twinr.proactive.runtime.portrait_match import derive_portrait_match
from twinr.proactive.runtime.speaker_association import ReSpeakerSpeakerAssociationSnapshot


class FakePortraitMatchProvider:
    def __init__(self, observation: PortraitMatchObservation) -> None:
        self.observation = observation
        self.backend = SimpleNamespace(name="fake_portrait_backend")
        self.calls = 0

    def observe(self) -> PortraitMatchObservation:
        self.calls += 1
        return self.observation


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

    def test_infers_single_visible_person_from_person_count_when_flag_is_missing(self) -> None:
        snapshot = derive_ambiguous_room_guard(
            observed_at=12.0,
            live_facts={
                "camera": {
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

        self.assertTrue(snapshot.person_visible)
        self.assertFalse(snapshot.guard_active)
        self.assertTrue(snapshot.clear)


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

    def test_likely_user_with_portrait_match_becomes_multimodal_hint(self) -> None:
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=20)).isoformat().replace("+00:00", "Z")
        portrait_snapshot = derive_portrait_match(
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
            provider=FakePortraitMatchProvider(
                PortraitMatchObservation(
                    checked_at=11.5,
                    state="likely_reference_user",
                    matches_reference_user=True,
                    confidence=0.88,
                    similarity_score=0.63,
                    live_face_count=1,
                    reference_face_count=1,
                    backend_name="fake_portrait_backend",
                )
            ),
            now_monotonic=12.0,
        )
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
            portrait_match=portrait_snapshot,
            now_utc=now_utc,
        )

        self.assertEqual(snapshot.state, "likely_main_user_multimodal")
        self.assertTrue(snapshot.matches_main_user)
        self.assertEqual(snapshot.policy_recommendation, "calm_personalization_only")
        self.assertEqual(snapshot.portrait_match_state, "likely_reference_user")
        self.assertIn("portrait_match", snapshot.claim.source)
        self.assertGreater(snapshot.claim.confidence, 0.84)

    def test_stable_temporal_portrait_match_upgrades_known_user_hint_source(self) -> None:
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=20)).isoformat().replace("+00:00", "Z")
        portrait_snapshot = derive_portrait_match(
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
            provider=FakePortraitMatchProvider(
                PortraitMatchObservation(
                    checked_at=11.5,
                    state="likely_reference_user",
                    matches_reference_user=True,
                    confidence=0.86,
                    fused_confidence=0.93,
                    temporal_state="stable_match",
                    temporal_observation_count=3,
                    similarity_score=0.66,
                    live_face_count=1,
                    reference_face_count=1,
                    reference_image_count=3,
                    matched_user_id="main_user",
                    backend_name="fake_portrait_backend",
                )
            ),
            now_monotonic=12.0,
        )
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
            portrait_match=portrait_snapshot,
            now_utc=now_utc,
        )

        self.assertEqual(snapshot.state, "likely_main_user_multimodal")
        self.assertEqual(snapshot.portrait_match_temporal_state, "stable_match")
        self.assertEqual(snapshot.portrait_match_observation_count, 3)
        self.assertEqual(
            snapshot.claim.source,
            "voice_profile_plus_temporal_portrait_match_plus_single_visible_person_context",
        )
        self.assertGreater(snapshot.claim.confidence, 0.87)

    def test_conflicting_portrait_match_forces_confirm_first(self) -> None:
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
            portrait_match=derive_portrait_match(
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
                provider=FakePortraitMatchProvider(
                    PortraitMatchObservation(
                        checked_at=11.5,
                        state="unknown_face",
                        matches_reference_user=False,
                        confidence=0.31,
                        similarity_score=0.21,
                        live_face_count=1,
                        reference_face_count=1,
                        backend_name="fake_portrait_backend",
                    )
                ),
                now_monotonic=12.0,
            ),
            now_utc=now_utc,
        )

        self.assertEqual(snapshot.state, "modality_conflict")
        self.assertFalse(snapshot.matches_main_user)
        self.assertEqual(snapshot.policy_recommendation, "confirm_first")
        self.assertEqual(snapshot.block_reason, "portrait_voice_mismatch")

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


class IdentityFusionTests(unittest.TestCase):
    def _base_live_facts(self) -> dict[str, object]:
        return {
            "camera": {
                "person_visible": True,
                "person_count": 1,
                "person_count_unknown": False,
                "primary_person_zone": "center",
                "looking_toward_device": True,
                "engaged_with_device": True,
                "visual_attention_score": 0.84,
            },
            "audio_policy": {
                "presence_audio_active": False,
                "recent_follow_up_speech": False,
                "resume_window_open": False,
                "speaker_direction_stable": True,
            },
            "vad": {
                "speech_detected": False,
            },
        }

    def _portrait_main_user_snapshot(self) -> object:
        return derive_portrait_match(
            observed_at=12.0,
            live_facts=self._base_live_facts(),
            provider=FakePortraitMatchProvider(
                PortraitMatchObservation(
                    checked_at=11.5,
                    state="likely_reference_user",
                    matches_reference_user=True,
                    confidence=0.88,
                    fused_confidence=0.93,
                    temporal_state="stable_match",
                    temporal_observation_count=3,
                    similarity_score=0.67,
                    live_face_count=1,
                    reference_face_count=1,
                    reference_image_count=3,
                    matched_user_id="main_user",
                    backend_name="fake_portrait_backend",
                )
            ),
            now_monotonic=12.0,
        )

    def test_tracker_builds_stable_main_user_multimodal_match(self) -> None:
        tracker = TemporalIdentityFusionTracker.from_config(TwinrConfig())
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=8)).isoformat().replace("+00:00", "Z")
        portrait_snapshot = self._portrait_main_user_snapshot()
        speaker_association = ReSpeakerSpeakerAssociationSnapshot(
            observed_at=12.0,
            state="primary_visible_person_associated",
            associated=True,
            target_id="primary_visible_person",
            confidence=0.88,
            camera_person_count=1,
            direction_confidence=0.89,
            azimuth_deg=0,
            primary_person_zone="center",
        )

        for observed_at in (10.0, 12.0, 14.0):
            snapshot = tracker.observe(
                observed_at=observed_at,
                live_facts=self._base_live_facts(),
                voice_status="likely_user",
                voice_confidence=0.84,
                voice_checked_at=checked_at,
                presence_session_id=7,
                portrait_match=portrait_snapshot,
                speaker_association=speaker_association,
                now_utc=now_utc,
            )

        self.assertEqual(snapshot.state, "stable_main_user_multimodal")
        self.assertTrue(snapshot.matches_main_user)
        self.assertEqual(snapshot.temporal_state, "stable_multimodal_match")
        self.assertEqual(snapshot.session_consistency_state, "stable_session")
        self.assertEqual(snapshot.track_consistency_state, "speaker_locked")
        self.assertEqual(
            snapshot.claim.source,
            "voice_profile_plus_temporal_portrait_match_plus_track_history_plus_presence_session_memory",
        )

    def test_known_user_hint_uses_temporal_identity_fusion_when_available(self) -> None:
        tracker = TemporalIdentityFusionTracker.from_config(TwinrConfig())
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=8)).isoformat().replace("+00:00", "Z")
        live_facts = self._base_live_facts()
        portrait_snapshot = self._portrait_main_user_snapshot()

        fusion_snapshot = None
        for observed_at in (10.0, 12.0, 14.0):
            fusion_snapshot = tracker.observe(
                observed_at=observed_at,
                live_facts=live_facts,
                voice_status="likely_user",
                voice_confidence=0.84,
                voice_checked_at=checked_at,
                presence_session_id=7,
                portrait_match=portrait_snapshot,
                now_utc=now_utc,
            )

        snapshot = derive_known_user_hint(
            observed_at=14.0,
            live_facts=live_facts,
            voice_status="likely_user",
            voice_confidence=0.84,
            voice_checked_at=checked_at,
            portrait_match=portrait_snapshot,
            identity_fusion=fusion_snapshot,
            now_utc=now_utc,
        )

        self.assertEqual(snapshot.state, "likely_main_user_temporal_multimodal")
        self.assertTrue(snapshot.matches_main_user)
        self.assertEqual(snapshot.identity_fusion_state, "stable_main_user_multimodal")
        self.assertEqual(
            snapshot.claim.source,
            "voice_profile_plus_temporal_portrait_match_plus_track_history_plus_presence_session_memory",
        )

    def test_tracker_blocks_other_enrolled_user_from_household_voice_signal(self) -> None:
        tracker = TemporalIdentityFusionTracker.from_config(TwinrConfig())
        now_utc = datetime(2026, 3, 19, 15, 0, tzinfo=timezone.utc)
        checked_at = (now_utc - timedelta(seconds=8)).isoformat().replace("+00:00", "Z")

        for observed_at in (10.0, 12.0, 14.0):
            snapshot = tracker.observe(
                observed_at=observed_at,
                live_facts=self._base_live_facts(),
                voice_status="known_other_user",
                voice_confidence=0.82,
                voice_checked_at=checked_at,
                voice_matched_user_id="guest_user",
                voice_matched_user_display_name="Guest",
                voice_match_source="household_voice_identity",
                presence_session_id=7,
                portrait_match=None,
                now_utc=now_utc,
            )

        self.assertEqual(snapshot.state, "stable_other_enrolled_user")
        self.assertFalse(snapshot.matches_main_user)
        self.assertEqual(snapshot.matched_user_id, "guest_user")
        self.assertEqual(snapshot.voice_matched_user_id, "guest_user")
        self.assertEqual(snapshot.policy_recommendation, "blocked")


class PortraitMatchTests(unittest.TestCase):
    def test_likely_reference_user_requires_clear_room_context(self) -> None:
        provider = FakePortraitMatchProvider(
            PortraitMatchObservation(
                checked_at=11.5,
                state="likely_reference_user",
                matches_reference_user=True,
                confidence=0.87,
                similarity_score=0.61,
                live_face_count=1,
                reference_face_count=1,
                backend_name="fake_portrait_backend",
            )
        )

        snapshot = derive_portrait_match(
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
            provider=provider,
            now_monotonic=12.0,
        )

        self.assertEqual(snapshot.state, "likely_reference_user")
        self.assertTrue(snapshot.matches_reference_user)
        self.assertEqual(snapshot.policy_recommendation, "calm_personalization_only")
        self.assertEqual(snapshot.backend_name, "fake_portrait_backend")
        self.assertEqual(provider.calls, 1)

    def test_stable_temporal_reference_user_uses_temporal_claim_source(self) -> None:
        provider = FakePortraitMatchProvider(
            PortraitMatchObservation(
                checked_at=11.5,
                state="likely_reference_user",
                matches_reference_user=True,
                confidence=0.86,
                fused_confidence=0.93,
                temporal_state="stable_match",
                temporal_observation_count=3,
                similarity_score=0.67,
                live_face_count=1,
                reference_face_count=1,
                reference_image_count=3,
                matched_user_id="main_user",
                backend_name="fake_portrait_backend",
            )
        )

        snapshot = derive_portrait_match(
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
            provider=provider,
            now_monotonic=12.0,
        )

        self.assertEqual(snapshot.state, "likely_reference_user")
        self.assertEqual(snapshot.temporal_state, "stable_match")
        self.assertEqual(snapshot.temporal_observation_count, 3)
        self.assertEqual(snapshot.claim.source, "local_portrait_match_temporal_fusion")
        self.assertGreater(snapshot.claim.confidence, 0.9)

    def test_other_enrolled_user_blocks_portrait_match(self) -> None:
        provider = FakePortraitMatchProvider(
            PortraitMatchObservation(
                checked_at=11.5,
                state="known_other_user",
                matches_reference_user=False,
                confidence=0.85,
                fused_confidence=0.9,
                temporal_state="stable_match",
                temporal_observation_count=2,
                similarity_score=0.62,
                live_face_count=1,
                reference_face_count=1,
                reference_image_count=2,
                matched_user_id="guest_user",
                matched_user_display_name="Guest",
                backend_name="fake_portrait_backend",
            )
        )

        snapshot = derive_portrait_match(
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
            provider=provider,
            now_monotonic=12.0,
        )

        self.assertEqual(snapshot.state, "known_other_user")
        self.assertEqual(snapshot.block_reason, "other_enrolled_user_detected")
        self.assertEqual(snapshot.policy_recommendation, "blocked")
        self.assertEqual(snapshot.matched_user_id, "guest_user")

    def test_ambiguous_room_blocks_portrait_match(self) -> None:
        snapshot = derive_portrait_match(
            observed_at=12.0,
            live_facts={
                "camera": {
                    "person_visible": True,
                    "person_count": 2,
                    "person_count_unknown": False,
                },
                "audio_policy": {},
            },
            provider=FakePortraitMatchProvider(
                PortraitMatchObservation(
                    checked_at=11.5,
                    state="likely_reference_user",
                    matches_reference_user=True,
                    confidence=0.87,
                    similarity_score=0.61,
                    live_face_count=1,
                    reference_face_count=1,
                    backend_name="fake_portrait_backend",
                )
            ),
            now_monotonic=12.0,
        )

        self.assertEqual(snapshot.state, "blocked_ambiguous_room")
        self.assertFalse(snapshot.matches_reference_user)
        self.assertEqual(snapshot.block_reason, "multi_person_context")


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
