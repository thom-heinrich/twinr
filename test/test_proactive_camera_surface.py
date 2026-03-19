from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.config import TwinrConfig
from twinr.proactive.runtime.service import ProactiveCoordinator
from twinr.proactive.social.camera_surface import ProactiveCameraSurface, ProactiveCameraSurfaceConfig
from twinr.proactive.social.engine import (
    SocialAudioObservation,
    SocialBodyPose,
    SocialDetectedObject,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialTriggerEngine,
    SocialVisionObservation,
)
from twinr.runtime import TwinrRuntime

from test.test_proactive_monitor import FakeAudioObserver, FakePirMonitor, FakeVisionObserver, MutableClock


class ProactiveCameraSurfaceTest(unittest.TestCase):
    def test_surface_emits_rising_edges_and_tracks_duration(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_event_cooldown_s=0.0,
                hand_or_object_near_camera_event_cooldown_s=0.0,
            )
        )

        first = surface.observe(
            inspected=True,
            observed_at=1.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=2,
                primary_person_zone=SocialPersonZone.CENTER,
                looking_toward_device=True,
                hand_or_object_near_camera=True,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        )
        second = surface.observe(
            inspected=True,
            observed_at=7.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=2,
                primary_person_zone=SocialPersonZone.CENTER,
                looking_toward_device=True,
                hand_or_object_near_camera=True,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        )

        self.assertEqual(
            first.event_names,
            (
                "camera.person_visible",
                "camera.hand_or_object_near_camera",
                "camera.attention_window_opened",
            ),
        )
        self.assertTrue(first.snapshot.person_visible)
        self.assertEqual(first.snapshot.person_count, 2)
        self.assertEqual(first.snapshot.primary_person_zone, SocialPersonZone.CENTER)
        self.assertTrue(first.snapshot.looking_toward_device)
        self.assertEqual(first.snapshot.body_pose, SocialBodyPose.UPRIGHT)
        self.assertEqual(second.event_names, ())
        self.assertGreaterEqual(second.snapshot.person_visible_for_s, 6.0)
        self.assertGreaterEqual(second.snapshot.hand_or_object_near_camera_for_s, 6.0)

    def test_surface_emits_fine_hand_gesture_event(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                gesture_event_cooldown_s=0.0,
            )
        )

        update = surface.observe(
            inspected=True,
            observed_at=2.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                body_pose=SocialBodyPose.UPRIGHT,
                fine_hand_gesture=SocialFineHandGesture.THUMBS_UP,
                fine_hand_gesture_confidence=0.88,
            ),
        )

        self.assertIn("camera.fine_hand_gesture_detected", update.event_names)
        self.assertEqual(update.snapshot.fine_hand_gesture, SocialFineHandGesture.THUMBS_UP)
        self.assertAlmostEqual(update.snapshot.fine_hand_gesture_confidence or 0.0, 0.88, places=3)

    def test_surface_holds_visible_state_through_brief_unknown_gap(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_unknown_hold_s=8.0,
                looking_toward_device_unknown_hold_s=8.0,
                hand_or_object_near_camera_unknown_hold_s=8.0,
                secondary_unknown_hold_s=8.0,
            )
        )
        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.LEFT,
                looking_toward_device=True,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        )

        update = surface.observe(
            inspected=False,
            observed_at=6.0,
            observation=SocialVisionObservation(person_visible=False),
        )

        self.assertTrue(update.snapshot.person_visible)
        self.assertTrue(update.snapshot.person_visible_unknown)
        self.assertEqual(update.snapshot.person_count, 1)
        self.assertTrue(update.snapshot.person_count_unknown)
        self.assertEqual(update.snapshot.primary_person_zone, SocialPersonZone.LEFT)
        self.assertTrue(update.snapshot.primary_person_zone_unknown)
        self.assertTrue(update.snapshot.looking_toward_device)
        self.assertTrue(update.snapshot.looking_toward_device_unknown)
        self.assertEqual(update.snapshot.body_pose, SocialBodyPose.UPRIGHT)
        self.assertTrue(update.snapshot.body_pose_unknown)
        self.assertEqual(update.event_names, ())

    def test_surface_fails_closed_after_unknown_hold_expires(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_unknown_hold_s=5.0,
                looking_toward_device_unknown_hold_s=5.0,
                hand_or_object_near_camera_unknown_hold_s=5.0,
                secondary_unknown_hold_s=5.0,
            )
        )
        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=2,
                primary_person_zone=SocialPersonZone.RIGHT,
                looking_toward_device=True,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        )

        update = surface.observe(
            inspected=False,
            observed_at=6.0,
            observation=SocialVisionObservation(person_visible=False),
        )

        self.assertFalse(update.snapshot.person_visible)
        self.assertTrue(update.snapshot.person_visible_unknown)
        self.assertEqual(update.snapshot.person_count, 0)
        self.assertTrue(update.snapshot.person_count_unknown)
        self.assertEqual(update.snapshot.primary_person_zone, SocialPersonZone.UNKNOWN)
        self.assertTrue(update.snapshot.primary_person_zone_unknown)
        self.assertFalse(update.snapshot.looking_toward_device)
        self.assertTrue(update.snapshot.looking_toward_device_unknown)
        self.assertEqual(update.snapshot.body_pose, SocialBodyPose.UNKNOWN)
        self.assertTrue(update.snapshot.body_pose_unknown)

    def test_surface_requires_two_negative_samples_to_clear_visibility(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_off_samples=2,
                hand_or_object_near_camera_off_samples=2,
            )
        )
        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )

        first_negative = surface.observe(
            inspected=True,
            observed_at=6.0,
            observation=SocialVisionObservation(person_visible=False),
        )
        second_negative = surface.observe(
            inspected=True,
            observed_at=12.0,
            observation=SocialVisionObservation(person_visible=False),
        )

        self.assertTrue(first_negative.snapshot.person_visible)
        self.assertFalse(second_negative.snapshot.person_visible)

    def test_surface_suppresses_fast_reactivation_events_with_cooldown(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_off_samples=1,
                person_visible_event_cooldown_s=10.0,
            )
        )

        first = surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )
        surface.observe(
            inspected=True,
            observed_at=2.0,
            observation=SocialVisionObservation(person_visible=False),
        )
        fast_return = surface.observe(
            inspected=True,
            observed_at=5.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )
        surface.observe(
            inspected=True,
            observed_at=11.0,
            observation=SocialVisionObservation(person_visible=False),
        )
        slow_return = surface.observe(
            inspected=True,
            observed_at=16.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )

        self.assertEqual(first.event_names, ("camera.person_visible",))
        self.assertEqual(fast_return.event_names, ())
        self.assertEqual(slow_return.event_names, ("camera.person_visible",))

    def test_surface_emits_person_returned_after_long_authoritative_absence(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_off_samples=1,
                person_visible_event_cooldown_s=0.0,
                person_returned_absence_s=15.0,
            )
        )

        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.LEFT,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        )
        surface.observe(
            inspected=True,
            observed_at=20.0,
            observation=SocialVisionObservation(person_visible=False),
        )
        returned = surface.observe(
            inspected=True,
            observed_at=40.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.RIGHT,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        )

        self.assertEqual(
            returned.event_names,
            ("camera.person_visible", "camera.person_returned"),
        )
        self.assertTrue(returned.snapshot.person_returned_after_absence)
        self.assertEqual(returned.snapshot.primary_person_zone, SocialPersonZone.RIGHT)

    def test_surface_does_not_emit_person_returned_after_unknown_only_gap(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_off_samples=1,
                person_visible_event_cooldown_s=0.0,
                person_visible_unknown_hold_s=5.0,
                person_returned_absence_s=3.0,
            )
        )

        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )
        surface.observe(
            inspected=False,
            observed_at=6.0,
            observation=SocialVisionObservation(person_visible=False),
        )
        returned = surface.observe(
            inspected=True,
            observed_at=8.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )

        self.assertEqual(returned.event_names, ("camera.person_visible",))
        self.assertFalse(returned.snapshot.person_returned_after_absence)

    def test_surface_projects_health_anchor_attention_gesture_and_object_signals(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_event_cooldown_s=0.0,
                hand_or_object_near_camera_event_cooldown_s=0.0,
                motion_event_cooldown_s=0.0,
                showing_intent_event_cooldown_s=0.0,
                gesture_event_cooldown_s=0.0,
                object_on_samples=1,
            )
        )

        update = surface.observe(
            inspected=True,
            observed_at=10.0,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=9.5,
                last_camera_health_change_at=9.0,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_box=SocialSpatialBox(top=0.1, left=0.2, bottom=0.8, right=0.7),
                looking_toward_device=True,
                person_near_device=True,
                engaged_with_device=True,
                visual_attention_score=0.83,
                body_pose=SocialBodyPose.SEATED,
                pose_confidence=0.72,
                motion_state=SocialMotionState.APPROACHING,
                motion_confidence=0.61,
                hand_or_object_near_camera=True,
                showing_intent_likely=True,
                coarse_arm_gesture=SocialGestureEvent.TIMEOUT_T,
                gesture_event=SocialGestureEvent.TIMEOUT_T,
                gesture_confidence=0.75,
                objects=(
                    SocialDetectedObject(
                        label="cup",
                        confidence=0.91,
                        zone=SocialPersonZone.RIGHT,
                        stable=False,
                        box=SocialSpatialBox(top=0.3, left=0.6, bottom=0.7, right=0.9),
                    ),
                ),
            ),
        )

        self.assertTrue(update.snapshot.camera_ready)
        self.assertTrue(update.snapshot.camera_ai_ready)
        self.assertEqual(update.snapshot.primary_person_zone, SocialPersonZone.CENTER)
        self.assertAlmostEqual(update.snapshot.primary_person_center_x or 0.0, 0.45, places=3)
        self.assertTrue(update.snapshot.person_near_device)
        self.assertTrue(update.snapshot.engaged_with_device)
        self.assertAlmostEqual(update.snapshot.visual_attention_score or 0.0, 0.83, places=3)
        self.assertEqual(update.snapshot.body_pose, SocialBodyPose.SEATED)
        self.assertAlmostEqual(update.snapshot.pose_confidence or 0.0, 0.72, places=3)
        self.assertEqual(update.snapshot.motion_state, SocialMotionState.APPROACHING)
        self.assertAlmostEqual(update.snapshot.motion_confidence or 0.0, 0.61, places=3)
        self.assertTrue(update.snapshot.showing_intent_likely)
        self.assertEqual(update.snapshot.coarse_arm_gesture, SocialGestureEvent.TIMEOUT_T)
        self.assertEqual(update.snapshot.gesture_event, SocialGestureEvent.TIMEOUT_T)
        self.assertEqual(len(update.snapshot.objects), 1)
        self.assertTrue(update.snapshot.objects[0].stable)
        self.assertEqual(
            update.event_names,
            (
                "camera.person_visible",
                "camera.hand_or_object_near_camera",
                "camera.attention_window_opened",
                "camera.showing_intent_started",
                "camera.motion_changed",
                "camera.gesture_detected",
                "camera.coarse_arm_gesture_detected",
                "camera.object_detected_stable",
            ),
        )

    def test_surface_holds_health_and_stable_objects_through_brief_unknown_gap(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                object_on_samples=1,
                secondary_unknown_hold_s=8.0,
                person_visible_unknown_hold_s=8.0,
                looking_toward_device_unknown_hold_s=8.0,
                person_near_device_unknown_hold_s=8.0,
                engaged_with_device_unknown_hold_s=8.0,
                showing_intent_unknown_hold_s=8.0,
                hand_or_object_near_camera_unknown_hold_s=8.0,
                object_unknown_hold_s=8.0,
            )
        )

        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.LEFT,
                primary_person_box=SocialSpatialBox(top=0.15, left=0.05, bottom=0.85, right=0.45),
                looking_toward_device=True,
                person_near_device=True,
                engaged_with_device=True,
                body_pose=SocialBodyPose.UPRIGHT,
                hand_or_object_near_camera=True,
                showing_intent_likely=True,
                objects=(
                    SocialDetectedObject(
                        label="medication_box",
                        confidence=0.88,
                        zone=SocialPersonZone.CENTER,
                        stable=False,
                    ),
                ),
            ),
        )

        update = surface.observe(
            inspected=False,
            observed_at=6.0,
            observation=SocialVisionObservation(),
        )

        self.assertTrue(update.snapshot.camera_ready)
        self.assertTrue(update.snapshot.camera_ready_unknown)
        self.assertTrue(update.snapshot.person_visible)
        self.assertTrue(update.snapshot.person_visible_unknown)
        self.assertTrue(update.snapshot.person_near_device)
        self.assertTrue(update.snapshot.person_near_device_unknown)
        self.assertTrue(update.snapshot.showing_intent_likely)
        self.assertTrue(update.snapshot.showing_intent_likely_unknown)
        self.assertEqual(len(update.snapshot.objects), 1)
        self.assertTrue(update.snapshot.objects_unknown)
        self.assertEqual(update.snapshot.objects[0].label, "medication_box")

    def test_surface_tracks_person_appearance_and_disappearance_timestamps(self) -> None:
        surface = ProactiveCameraSurface(
            config=ProactiveCameraSurfaceConfig(
                person_visible_off_samples=1,
                person_visible_event_cooldown_s=0.0,
            )
        )

        appeared = surface.observe(
            inspected=True,
            observed_at=3.0,
            observation=SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
        )
        disappeared = surface.observe(
            inspected=True,
            observed_at=9.0,
            observation=SocialVisionObservation(person_visible=False),
        )
        repeated_absence = surface.observe(
            inspected=True,
            observed_at=15.0,
            observation=SocialVisionObservation(person_visible=False),
        )

        self.assertEqual(appeared.snapshot.person_appeared_at, 3.0)
        self.assertEqual(disappeared.snapshot.person_disappeared_at, 9.0)
        self.assertEqual(repeated_absence.snapshot.person_disappeared_at, 9.0)


class ProactiveCameraCoordinatorIntegrationTest(unittest.TestCase):
    def test_coordinator_keeps_camera_presence_facts_during_uninspected_gap(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr-camera-surface-test",
            proactive_enabled=True,
            proactive_capture_interval_s=10.0,
            proactive_motion_window_s=20.0,
        )
        runtime = TwinrRuntime(config=config)
        clock = MutableClock(0.0)
        observations: list[tuple[dict[str, object], tuple[str, ...]]] = []
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [
                    SocialVisionObservation(
                        person_visible=True,
                        person_count=2,
                        primary_person_zone=SocialPersonZone.CENTER,
                        looking_toward_device=True,
                        body_pose=SocialBodyPose.UPRIGHT,
                    )
                ]
            ),
            pir_monitor=FakePirMonitor(events=[True], level=True),
            audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
            observation_handler=lambda facts, event_names: observations.append((facts, event_names)),
            emit=lambda _line: None,
            clock=clock,
        )

        coordinator.tick()
        clock.now = 5.0
        coordinator.tick()

        self.assertEqual(len(observations), 2)
        first_facts, first_events = observations[0]
        second_facts, second_events = observations[1]
        self.assertIn("camera.person_visible", first_events)
        self.assertTrue(first_facts["camera"]["person_visible"])
        self.assertEqual(first_facts["camera"]["person_count"], 2)
        self.assertEqual(first_facts["camera"]["count_persons"], 2)
        self.assertEqual(first_facts["camera"]["primary_person_zone"], "center")
        self.assertTrue(second_facts["camera"]["person_visible"])
        self.assertTrue(second_facts["camera"]["person_visible_unknown"])
        self.assertTrue(second_facts["camera"]["person_count_unknown"])
        self.assertTrue(second_facts["camera"]["primary_person_zone_unknown"])
        self.assertEqual(second_events, ())

    def test_coordinator_emits_camera_person_returned_event(self) -> None:
        config = TwinrConfig(
            project_root="/tmp/twinr-camera-return-test",
            proactive_enabled=True,
            proactive_capture_interval_s=1.0,
            proactive_motion_window_s=20.0,
            proactive_person_returned_absence_s=30.0,
        )
        runtime = TwinrRuntime(config=config)
        clock = MutableClock(0.0)
        observations: list[tuple[dict[str, object], tuple[str, ...]]] = []
        pir_monitor = FakePirMonitor(events=[True], level=True)
        coordinator = ProactiveCoordinator(
            config=config,
            runtime=runtime,
            engine=SocialTriggerEngine(),
            trigger_handler=lambda _decision: True,
            vision_observer=FakeVisionObserver(
                [
                    SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                    SocialVisionObservation(person_visible=False),
                    SocialVisionObservation(person_visible=False),
                    SocialVisionObservation(person_visible=True, body_pose=SocialBodyPose.UPRIGHT),
                ]
            ),
            pir_monitor=pir_monitor,
            audio_observer=FakeAudioObserver(SocialAudioObservation(speech_detected=False)),
            observation_handler=lambda facts, event_names: observations.append((facts, event_names)),
            emit=lambda _line: None,
            clock=clock,
        )

        coordinator.tick()
        clock.now = 40.0
        pir_monitor.events = [True]
        coordinator.tick()
        clock.now = 41.0
        pir_monitor.events = [True]
        coordinator.tick()
        clock.now = 80.0
        pir_monitor.events = [True]
        coordinator.tick()

        self.assertEqual(len(observations), 4)
        returned_facts, returned_events = observations[-1]
        self.assertIn("camera.person_returned", returned_events)
        self.assertTrue(returned_facts["camera"]["person_returned_after_absence"])


if __name__ == "__main__":
    unittest.main()
