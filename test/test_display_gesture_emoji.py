from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.emoji_cues import DisplayEmojiCue
from twinr.proactive.runtime.display_gesture_emoji import (
    DisplayGestureEmojiPublisher,
    display_gesture_refresh_supported,
    derive_display_gesture_emoji,
    resolve_display_gesture_refresh_interval,
)
from twinr.proactive.social.camera_surface import ProactiveCameraSnapshot, ProactiveCameraSurfaceUpdate
from twinr.proactive.social.engine import (
    SocialBodyPose,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
)


def _snapshot(
    *,
    fine_hand_gesture: SocialFineHandGesture = SocialFineHandGesture.NONE,
    fine_hand_gesture_unknown: bool = False,
    gesture_event: SocialGestureEvent = SocialGestureEvent.NONE,
    gesture_event_unknown: bool = False,
) -> ProactiveCameraSnapshot:
    return ProactiveCameraSnapshot(
        camera_online=True,
        camera_online_unknown=False,
        camera_ready=True,
        camera_ready_unknown=False,
        camera_ai_ready=True,
        camera_ai_ready_unknown=False,
        camera_error=None,
        camera_error_unknown=False,
        last_camera_frame_at=10.0,
        last_camera_frame_at_unknown=False,
        last_camera_health_change_at=10.0,
        last_camera_health_change_at_unknown=False,
        person_visible=True,
        person_visible_for_s=1.0,
        person_visible_unknown=False,
        person_recently_visible=True,
        person_recently_visible_unknown=False,
        person_count=1,
        person_count_unknown=False,
        person_appeared_at=9.0,
        person_appeared_at_unknown=False,
        person_disappeared_at=None,
        person_disappeared_at_unknown=False,
        person_returned_after_absence=False,
        primary_person_zone=SocialPersonZone.CENTER,
        primary_person_zone_unknown=False,
        primary_person_box=None,
        primary_person_box_unknown=False,
        primary_person_center_x=0.5,
        primary_person_center_x_unknown=False,
        primary_person_center_y=0.5,
        primary_person_center_y_unknown=False,
        looking_toward_device=True,
        looking_toward_device_unknown=False,
        person_near_device=True,
        person_near_device_unknown=False,
        engaged_with_device=True,
        engaged_with_device_unknown=False,
        visual_attention_score=0.9,
        visual_attention_score_unknown=False,
        body_pose=SocialBodyPose.UPRIGHT,
        body_pose_unknown=False,
        pose_confidence=0.9,
        pose_confidence_unknown=False,
        body_state_changed_at=9.0,
        body_state_changed_at_unknown=False,
        motion_state=SocialMotionState.STILL,
        motion_state_unknown=False,
        motion_confidence=0.8,
        motion_confidence_unknown=False,
        motion_state_changed_at=9.0,
        motion_state_changed_at_unknown=False,
        smiling=False,
        smiling_unknown=False,
        hand_or_object_near_camera=True,
        hand_or_object_near_camera_for_s=1.0,
        hand_or_object_near_camera_unknown=False,
        showing_intent_likely=True,
        showing_intent_likely_unknown=False,
        showing_intent_started_at=9.0,
        showing_intent_started_at_unknown=False,
        gesture_event=gesture_event,
        gesture_event_unknown=gesture_event_unknown,
        gesture_confidence=0.84,
        gesture_confidence_unknown=False,
        fine_hand_gesture=fine_hand_gesture,
        fine_hand_gesture_unknown=fine_hand_gesture_unknown,
        fine_hand_gesture_confidence=0.91,
        fine_hand_gesture_confidence_unknown=False,
        objects=(),
        objects_unknown=False,
    )


class DisplayGestureEmojiTests(unittest.TestCase):
    def test_refresh_interval_uses_dedicated_gesture_setting_instead_of_attention_cadence(self) -> None:
        config = TwinrConfig(
            project_root=".",
            display_attention_refresh_interval_s=0.6,
            display_gesture_refresh_interval_s=0.2,
        )

        interval_s = resolve_display_gesture_refresh_interval(config)

        self.assertEqual(interval_s, 0.2)

    def test_refresh_interval_fails_closed_when_voice_orchestrator_is_enabled(self) -> None:
        config = TwinrConfig(
            project_root=".",
            display_gesture_refresh_interval_s=0.2,
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://example.invalid/ws/orchestrator/voice",
        )

        interval_s = resolve_display_gesture_refresh_interval(config)

        self.assertIsNone(interval_s)

    def test_gesture_refresh_support_fails_closed_when_voice_orchestrator_is_enabled(self) -> None:
        config = TwinrConfig(
            project_root=".",
            display_driver="hdmi_wayland",
            display_gesture_refresh_interval_s=0.2,
            voice_orchestrator_enabled=True,
            voice_orchestrator_ws_url="ws://example.invalid/ws/orchestrator/voice",
        )

        supported = display_gesture_refresh_supported(
            config=config,
            vision_observer=type("_VisionObserver", (), {"supports_gesture_refresh": True})(),
        )

        self.assertFalse(supported)

    def test_derive_prefers_fine_hand_gesture_symbols(self) -> None:
        decision = derive_display_gesture_emoji(
            snapshot=_snapshot(fine_hand_gesture=SocialFineHandGesture.THUMBS_UP),
            event_names=("camera.fine_hand_gesture_detected",),
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "thumbs_up")
        self.assertEqual(decision.accent, "success")

    def test_derive_ignores_unsupported_coarse_gestures(self) -> None:
        decision = derive_display_gesture_emoji(
            snapshot=_snapshot(gesture_event=SocialGestureEvent.WAVE),
            event_names=("camera.gesture_detected",),
        )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "unsupported_coarse_gesture")

    def test_derive_ignores_open_palm_even_when_wave_event_is_also_present(self) -> None:
        decision = derive_display_gesture_emoji(
            snapshot=_snapshot(
                gesture_event=SocialGestureEvent.WAVE,
                fine_hand_gesture=SocialFineHandGesture.OPEN_PALM,
            ),
            event_names=("camera.fine_hand_gesture_detected", "camera.gesture_detected"),
        )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "unsupported_fine_hand_gesture")

    def test_derive_ignores_unsupported_or_hostile_gestures(self) -> None:
        decision = derive_display_gesture_emoji(
            snapshot=_snapshot(fine_hand_gesture=SocialFineHandGesture.MIDDLE_FINGER),
            event_names=("camera.fine_hand_gesture_detected",),
        )

        self.assertFalse(decision.active)
        self.assertEqual(decision.reason, "unsupported_fine_hand_gesture")

    def test_derive_maps_peace_sign_to_victory_hand(self) -> None:
        decision = derive_display_gesture_emoji(
            snapshot=_snapshot(fine_hand_gesture=SocialFineHandGesture.PEACE_SIGN),
            event_names=("camera.fine_hand_gesture_detected",),
        )

        self.assertTrue(decision.active)
        self.assertEqual(decision.symbol.value, "victory_hand")
        self.assertEqual(decision.accent, "warm")

    def test_publisher_does_not_overwrite_foreign_emoji_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayGestureEmojiPublisher.from_config(config)
            publisher.store.save(
                DisplayEmojiCue(source="presentation", symbol="sparkles", accent="info"),
                hold_seconds=10.0,
                now=datetime(2026, 3, 20, 18, 0, tzinfo=timezone.utc),
            )

            result = publisher.publish_update(
                ProactiveCameraSurfaceUpdate(
                    snapshot=_snapshot(fine_hand_gesture=SocialFineHandGesture.THUMBS_UP),
                    event_names=("camera.fine_hand_gesture_detected",),
                ),
                now=datetime(2026, 3, 20, 18, 0, 1, tzinfo=timezone.utc),
            )
            active = publisher.store.load_active(now=datetime(2026, 3, 20, 18, 0, 2, tzinfo=timezone.utc))

        self.assertEqual(result.action, "blocked_foreign_cue")
        self.assertIsNotNone(active)
        assert active is not None
        self.assertEqual(active.source, "presentation")
        self.assertEqual(active.symbol, "sparkles")

    def test_publisher_ignores_unsupported_wave_and_ok_sign_without_persisting_cue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayGestureEmojiPublisher.from_config(config)

            wave_result = publisher.publish_update(
                ProactiveCameraSurfaceUpdate(
                    snapshot=_snapshot(gesture_event=SocialGestureEvent.WAVE),
                    event_names=("camera.gesture_detected",),
                ),
                now=datetime(2026, 3, 21, 12, 0, 0, tzinfo=timezone.utc),
            )
            ok_result = publisher.publish_update(
                ProactiveCameraSurfaceUpdate(
                    snapshot=_snapshot(fine_hand_gesture=SocialFineHandGesture.OK_SIGN),
                    event_names=("camera.fine_hand_gesture_detected",),
                ),
                now=datetime(2026, 3, 21, 12, 0, 0, 400000, tzinfo=timezone.utc),
            )
            active = publisher.store.load_active(now=datetime(2026, 3, 21, 12, 0, 0, 500000, tzinfo=timezone.utc))

        self.assertEqual(wave_result.action, "inactive")
        self.assertEqual(wave_result.decision.reason, "unsupported_coarse_gesture")
        self.assertEqual(ok_result.action, "inactive")
        self.assertEqual(ok_result.decision.reason, "unsupported_fine_hand_gesture")
        self.assertIsNone(active)


if __name__ == "__main__":
    unittest.main()
