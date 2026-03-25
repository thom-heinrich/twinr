from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twinr.agent.base_agent import TwinrConfig
from twinr.display.debug_signals import DisplayDebugSignalStore
from twinr.proactive.runtime.display_debug_signals import DisplayDebugSignalPublisher


class DisplayDebugSignalPublisherTests(unittest.TestCase):
    def test_publisher_maps_current_camera_facts_into_header_signals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayDebugSignalPublisher.from_config(config)
            now = datetime(2026, 3, 23, 11, 10, tzinfo=timezone.utc)

            result = publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 2,
                    "person_count_unknown": False,
                    "looking_toward_device": True,
                    "looking_toward_device_unknown": False,
                    "looking_signal_state": "confirmed",
                    "hand_or_object_near_camera": True,
                    "hand_or_object_near_camera_unknown": False,
                    "showing_intent_likely": True,
                    "showing_intent_likely_unknown": False,
                    "engaged_with_device": True,
                    "engaged_with_device_unknown": False,
                    "body_pose": "upright",
                    "body_pose_unknown": False,
                    "motion_state": "still",
                    "motion_state_unknown": False,
                },
                now=now,
            )
            snapshot = DisplayDebugSignalStore.from_config(config).load_active(now=now)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(result.action, "updated")
        self.assertEqual(
            tuple(signal.label for signal in snapshot.signals),
            (
                "LOOKING_CONFIRMED",
                "HAND_NEAR",
                "INTENT_LIKELY",
                "ENGAGED",
                "PERSON_2",
                "POSE_UPRIGHT",
                "MOTION_STILL",
            ),
        )
        self.assertEqual(
            result.signal_keys,
            (
                "looking_toward_device",
                "hand_or_object_near_camera",
                "showing_intent_likely",
                "engaged_with_device",
                "person_visible",
                "body_pose",
                "motion_state",
            ),
        )

    def test_publisher_marks_engaged_without_looking_as_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayDebugSignalPublisher.from_config(config)
            now = datetime(2026, 3, 23, 11, 10, tzinfo=timezone.utc)

            publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 1,
                    "person_count_unknown": False,
                    "engaged_with_device": True,
                    "engaged_with_device_unknown": False,
                    "looking_toward_device": False,
                    "looking_toward_device_unknown": False,
                },
                now=now,
            )
            snapshot = DisplayDebugSignalStore.from_config(config).load_active(now=now)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(tuple(signal.label for signal in snapshot.signals), ("ENGAGED_PROXY", "PERSON_1"))

    def test_publisher_renders_proxy_looking_pill_when_only_proxy_state_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayDebugSignalPublisher.from_config(config)
            now = datetime(2026, 3, 23, 11, 10, tzinfo=timezone.utc)

            publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 1,
                    "person_count_unknown": False,
                    "looking_toward_device": True,
                    "looking_toward_device_unknown": False,
                    "looking_signal_state": "proxy",
                },
                now=now,
            )
            snapshot = DisplayDebugSignalStore.from_config(config).load_active(now=now)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(tuple(signal.label for signal in snapshot.signals), ("LOOKING_PROXY", "PERSON_1"))

    def test_publisher_keeps_engaged_as_proxy_when_looking_is_only_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayDebugSignalPublisher.from_config(config)
            now = datetime(2026, 3, 23, 11, 10, tzinfo=timezone.utc)

            publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 1,
                    "person_count_unknown": False,
                    "engaged_with_device": True,
                    "engaged_with_device_unknown": False,
                    "looking_toward_device": True,
                    "looking_toward_device_unknown": False,
                    "looking_signal_state": "proxy",
                },
                now=now,
            )
            snapshot = DisplayDebugSignalStore.from_config(config).load_active(now=now)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(tuple(signal.label for signal in snapshot.signals), ("LOOKING_PROXY", "ENGAGED_PROXY", "PERSON_1"))

    def test_publisher_holds_recent_event_and_safety_trigger_signals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            publisher = DisplayDebugSignalPublisher.from_config(config)
            first_now = datetime(2026, 3, 23, 11, 10, tzinfo=timezone.utc)
            publisher.publish_from_camera_facts(
                camera_facts={},
                event_names=("camera.attention_window_opened", "camera.person_returned"),
                trigger_ids=("possible_fall", "distress_possible", "positive_contact"),
                now=first_now,
            )

            held_now = first_now + timedelta(seconds=4)
            publisher.publish_from_camera_facts(
                camera_facts={},
                now=held_now,
            )
            held_snapshot = DisplayDebugSignalStore.from_config(config).load_active(now=held_now)

            expired_now = first_now + timedelta(seconds=11)
            publisher.publish_from_camera_facts(
                camera_facts={},
                now=expired_now,
            )
            expired_snapshot = DisplayDebugSignalStore.from_config(config).load_active(now=expired_now)

        self.assertIsNotNone(held_snapshot)
        assert held_snapshot is not None
        self.assertEqual(
            tuple(signal.label for signal in held_snapshot.signals),
            (
                "POSSIBLE_FALL",
                "DISTRESS_POSSIBLE",
                "ATTENTION_WINDOW",
                "POSITIVE_CONTACT",
                "PERSON_RETURNED",
            ),
        )
        self.assertIsNotNone(expired_snapshot)
        assert expired_snapshot is not None
        self.assertEqual(expired_snapshot.signals, ())

    def test_publisher_skips_unchanged_rewrites_until_snapshot_nears_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TwinrConfig(project_root=temp_dir)
            store = DisplayDebugSignalStore.from_config(config)
            publisher = DisplayDebugSignalPublisher.from_config(config)
            first_now = datetime(2026, 3, 23, 11, 10, tzinfo=timezone.utc)
            first = publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 1,
                    "person_count_unknown": False,
                },
                now=first_now,
            )
            first_snapshot = store.load_active(now=first_now)
            assert first_snapshot is not None

            unchanged_now = first_now + timedelta(seconds=2)
            unchanged = publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 1,
                    "person_count_unknown": False,
                },
                now=unchanged_now,
            )
            unchanged_snapshot = store.load_active(now=unchanged_now)
            assert unchanged_snapshot is not None

            refresh_now = first_now + timedelta(seconds=4.7)
            refreshed = publisher.publish_from_camera_facts(
                camera_facts={
                    "person_visible": True,
                    "person_visible_unknown": False,
                    "person_count": 1,
                    "person_count_unknown": False,
                },
                now=refresh_now,
            )
            refreshed_snapshot = store.load_active(now=refresh_now)
            assert refreshed_snapshot is not None

        self.assertEqual(first.action, "updated")
        self.assertEqual(unchanged.action, "unchanged")
        self.assertEqual(refreshed.action, "refreshed")
        self.assertEqual(unchanged_snapshot.updated_at, first_snapshot.updated_at)
        self.assertNotEqual(refreshed_snapshot.updated_at, first_snapshot.updated_at)
        self.assertEqual(refreshed_snapshot.signature(), first_snapshot.signature())


if __name__ == "__main__":
    unittest.main()
