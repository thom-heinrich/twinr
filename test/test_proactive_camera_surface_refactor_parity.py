from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.proactive.social.camera_surface import ProactiveCameraSurface, ProactiveCameraSurfaceConfig
from twinr.proactive.social.engine import (
    SocialBodyPose,
    SocialDetectedObject,
    SocialFineHandGesture,
    SocialGestureEvent,
    SocialMotionState,
    SocialPersonZone,
    SocialSpatialBox,
    SocialVisionObservation,
)
from twinr.proactive.social.gesture_calibration import FineHandGesturePolicy, GestureCalibrationProfile

try:
    from twinr.proactive.social.camera_surface_impl import ProactiveCameraSurfaceImpl
except ImportError:  # pragma: no cover - exercised after the refactor package exists
    ProactiveCameraSurfaceImpl = None

_EXPECTED_GOLDEN_DIGESTS = {
    "from_config_fast_attention": "2433b18e17344e2ddcb113cef9f3e140cf8c8e73c3378f7f9642d06bd4d9b4c2",
    "mixed_presence_sequence": "4458ca679a2afbd29f2bd27ef6b68a575bc44f3eaa5bd5eb08a5ab023b8a404c",
    "gesture_and_object_sequence": "daf941e27ea4878f20502762c159169d7890cb7cd9039704ecfcd9c2b3093b91",
    "person_return_sequence": "cfe33f67d9aeaaa38ffda4b7ca32fd86da9f67458a655c83eaf67573088d50ca",
}


def _payload_digest(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _update_payload(update) -> dict[str, object]:
    return {
        "event_names": list(update.event_names),
        "snapshot": update.snapshot.to_automation_facts(),
    }


def _capture_from_config_payload() -> dict[str, object]:
    config = ProactiveCameraSurfaceConfig.from_config(
        SimpleNamespace(
            proactive_capture_interval_s=6.0,
            display_attention_refresh_interval_s=0.35,
            proactive_person_returned_absence_s=123.0,
            proactive_local_camera_fine_hand_explicit_hold_s=0.42,
            proactive_local_camera_fine_hand_explicit_confirm_samples=2,
            proactive_local_camera_fine_hand_explicit_min_confidence=0.74,
        )
    )
    return asdict(config)


def _capture_mixed_presence_sequence(
    surface_cls: type[ProactiveCameraSurface] = ProactiveCameraSurface,
) -> list[dict[str, object]]:
    surface = surface_cls(
        config=ProactiveCameraSurfaceConfig(
            person_visible_event_cooldown_s=0.0,
            hand_or_object_near_camera_event_cooldown_s=0.0,
            person_visible_unknown_hold_s=5.0,
            looking_toward_device_unknown_hold_s=5.0,
            hand_or_object_near_camera_unknown_hold_s=5.0,
            secondary_unknown_hold_s=5.0,
            primary_person_center_smoothing_alpha=0.5,
            primary_person_center_deadband=0.02,
            primary_person_center_smoothing_window_s=2.0,
        )
    )
    steps = (
        surface.observe(
            inspected=True,
            observed_at=1.0,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=1.0,
                person_visible=True,
                person_count=2,
                primary_person_zone=SocialPersonZone.LEFT,
                primary_person_box=SocialSpatialBox(top=0.1, left=0.18, bottom=0.85, right=0.48),
                looking_toward_device=True,
                hand_or_object_near_camera=True,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        ),
        surface.observe(
            inspected=False,
            observed_at=4.0,
            observation=SocialVisionObservation(person_visible=False),
        ),
        surface.observe(
            inspected=True,
            observed_at=4.6,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=False,
                camera_ai_ready=False,
                camera_error="camera_busy",
                last_camera_frame_at=1.0,
                person_visible=False,
                person_count=0,
            ),
        ),
        surface.observe(
            inspected=True,
            observed_at=7.5,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=7.5,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_box=SocialSpatialBox(top=0.12, left=0.3, bottom=0.84, right=0.62),
                looking_toward_device=True,
                hand_or_object_near_camera=False,
                body_pose=SocialBodyPose.SEATED,
            ),
        ),
    )
    return [_update_payload(step) for step in steps]


def _capture_gesture_and_object_sequence(
    surface_cls: type[ProactiveCameraSurface] = ProactiveCameraSurface,
) -> list[dict[str, object]]:
    surface = surface_cls(
        config=ProactiveCameraSurfaceConfig(
            person_visible_event_cooldown_s=0.0,
            hand_or_object_near_camera_event_cooldown_s=0.0,
            motion_event_cooldown_s=0.0,
            showing_intent_event_cooldown_s=0.0,
            gesture_event_cooldown_s=0.0,
            object_on_samples=1,
            fine_hand_explicit_confirm_samples=2,
            fine_hand_explicit_hold_s=0.35,
            fine_hand_explicit_min_confidence=0.7,
            gesture_calibration=GestureCalibrationProfile(
                fine_hand={
                    SocialFineHandGesture.OK_SIGN: FineHandGesturePolicy(
                        min_confidence=0.8,
                        confirm_samples=2,
                        hold_s=0.35,
                    ),
                }
            ),
        )
    )
    steps = (
        surface.observe(
            inspected=True,
            observed_at=10.0,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=9.8,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_box=SocialSpatialBox(top=0.1, left=0.2, bottom=0.8, right=0.68),
                person_near_device=True,
                engaged_with_device=True,
                visual_attention_score=0.82,
                body_pose=SocialBodyPose.SEATED,
                pose_confidence=0.71,
                motion_state=SocialMotionState.APPROACHING,
                motion_confidence=0.63,
                hand_or_object_near_camera=True,
                showing_intent_likely=True,
                gesture_event=SocialGestureEvent.WAVE,
                gesture_confidence=0.74,
                fine_hand_gesture=SocialFineHandGesture.OK_SIGN,
                fine_hand_gesture_confidence=0.84,
                objects=(
                    SocialDetectedObject(
                        label="cup",
                        confidence=0.91,
                        zone=SocialPersonZone.RIGHT,
                        stable=False,
                        box=SocialSpatialBox(top=0.32, left=0.58, bottom=0.72, right=0.9),
                    ),
                ),
            ),
        ),
        surface.observe(
            inspected=True,
            observed_at=10.2,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=10.2,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                primary_person_box=SocialSpatialBox(top=0.1, left=0.21, bottom=0.8, right=0.69),
                person_near_device=True,
                engaged_with_device=True,
                visual_attention_score=0.83,
                body_pose=SocialBodyPose.SEATED,
                pose_confidence=0.73,
                motion_state=SocialMotionState.APPROACHING,
                motion_confidence=0.66,
                hand_or_object_near_camera=True,
                showing_intent_likely=True,
                coarse_arm_gesture=SocialGestureEvent.STOP,
                gesture_event=SocialGestureEvent.WAVE,
                gesture_confidence=0.77,
                fine_hand_gesture=SocialFineHandGesture.OK_SIGN,
                fine_hand_gesture_confidence=0.86,
                objects=(
                    SocialDetectedObject(
                        label="cup",
                        confidence=0.89,
                        zone=SocialPersonZone.RIGHT,
                        stable=False,
                        box=SocialSpatialBox(top=0.31, left=0.57, bottom=0.73, right=0.91),
                    ),
                ),
            ),
        ),
        surface.observe(
            inspected=True,
            observed_at=10.45,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=10.45,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                body_pose=SocialBodyPose.SEATED,
                fine_hand_gesture=SocialFineHandGesture.NONE,
                fine_hand_gesture_confidence=None,
                gesture_event=SocialGestureEvent.NONE,
                objects=(),
            ),
        ),
        surface.observe(
            inspected=True,
            observed_at=10.9,
            observation=SocialVisionObservation(
                camera_online=True,
                camera_ready=True,
                camera_ai_ready=True,
                last_camera_frame_at=10.9,
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.CENTER,
                body_pose=SocialBodyPose.SEATED,
                fine_hand_gesture=SocialFineHandGesture.NONE,
                fine_hand_gesture_confidence=None,
                gesture_event=SocialGestureEvent.NONE,
                objects=(),
            ),
        ),
    )
    return [_update_payload(step) for step in steps]


def _capture_person_return_sequence(
    surface_cls: type[ProactiveCameraSurface] = ProactiveCameraSurface,
) -> list[dict[str, object]]:
    surface = surface_cls(
        config=ProactiveCameraSurfaceConfig(
            person_visible_off_samples=1,
            person_visible_event_cooldown_s=0.0,
            person_returned_absence_s=15.0,
        )
    )
    steps = (
        surface.observe(
            inspected=True,
            observed_at=0.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.LEFT,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        ),
        surface.observe(
            inspected=True,
            observed_at=20.0,
            observation=SocialVisionObservation(person_visible=False),
        ),
        surface.observe(
            inspected=True,
            observed_at=40.0,
            observation=SocialVisionObservation(
                person_visible=True,
                person_count=1,
                primary_person_zone=SocialPersonZone.RIGHT,
                body_pose=SocialBodyPose.UPRIGHT,
            ),
        ),
    )
    return [_update_payload(step) for step in steps]


class ProactiveCameraSurfaceRefactorParityTests(unittest.TestCase):
    def test_public_wrapper_preserves_class_module(self) -> None:
        self.assertEqual(
            ProactiveCameraSurface.__module__,
            "twinr.proactive.social.camera_surface",
        )

    def test_golden_master_hashes_remain_stable(self) -> None:
        cases = {
            "from_config_fast_attention": _capture_from_config_payload(),
            "mixed_presence_sequence": _capture_mixed_presence_sequence(),
            "gesture_and_object_sequence": _capture_gesture_and_object_sequence(),
            "person_return_sequence": _capture_person_return_sequence(),
        }
        for name, payload in cases.items():
            with self.subTest(case=name):
                self.assertEqual(_payload_digest(payload), _EXPECTED_GOLDEN_DIGESTS[name])

    @unittest.skipIf(
        ProactiveCameraSurfaceImpl is None,
        "internal refactor package not present yet",
    )
    def test_public_wrapper_matches_internal_implementation_payloads(self) -> None:
        builders = (
            ("mixed_presence_sequence", _capture_mixed_presence_sequence),
            ("gesture_and_object_sequence", _capture_gesture_and_object_sequence),
            ("person_return_sequence", _capture_person_return_sequence),
        )
        for name, builder in builders:
            with self.subTest(case=name):
                wrapped = builder(ProactiveCameraSurface)
                internal = builder(ProactiveCameraSurfaceImpl)
                self.assertEqual(wrapped, internal)
